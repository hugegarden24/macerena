import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import "./App.css";

const PRESETS = {
  Pyramid: () => new THREE.ConeGeometry(1.4, 2, 4).rotateY(Math.PI / 4),
  Cube: () => new THREE.BoxGeometry(1.7, 1.7, 1.7),
  Cone: () => new THREE.ConeGeometry(1.1, 2.2, 48),
  Sphere: () => new THREE.SphereGeometry(1.2, 28, 18),
  Cylinder: () => new THREE.CylinderGeometry(0.9, 0.9, 2, 36),
  Torus: () => new THREE.TorusGeometry(1, 0.35, 20, 64),
  "Torus Knot": () => new THREE.TorusKnotGeometry(0.9, 0.3, 160, 24),
  Octahedron: () => new THREE.OctahedronGeometry(1.4),
  Dodecahedron: () => new THREE.DodecahedronGeometry(1.3),
  Icosahedron: () => new THREE.IcosahedronGeometry(1.3),
};

const MODES = [
  ["wire", "Wire"],
  ["solid", "Solid"],
  ["original", "Original"],
];

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
  [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const PINCH_THRESHOLD = 0.05;
const isFist = (l) =>
  [[8, 6], [12, 10], [16, 14], [20, 18]].every(
    ([tip, pip]) => dist(l[tip], l[0]) < dist(l[pip], l[0])
  );

// how far parts fly apart (bigger = more exploded)
const EXPLODE_SPREAD = 1.8; // pushes each part away along its own position
const EXPLODE_PUSH = 0.25; // extra push along its direction, as a fraction of model size

export default function App() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const mountRef = useRef(null);
  const fileRef = useRef(null);
  const api = useRef(null);

  const [status, setStatus] = useState("Loading hand tracker...");
  const [active, setActive] = useState("Pyramid");
  const [mode, setModeState] = useState("wire");
  const [error, setError] = useState("");
  const [custom, setCustom] = useState(false);
  const [partCount, setPartCount] = useState(0);
  const [exploded, setExploded] = useState(false);
  const [selName, setSelName] = useState("");

  useEffect(() => {
    let raf, stream, landmarker;
    let cancelled = false;
    let lastTime = -1;
    let lastLabel = "";
    let current = null;
    let viewMode = "wire";
    let parts = [];
    let selected = null; // permanently selected part (stays until deselected)
    let drag = null; // { part, src: "hand" | "mouse" } while actively moving
    const ex = { t: 0, v: 0 };
    const ray = new THREE.Raycaster();
    const mouse = { mode: null, x: 0, y: 0, dist: 0 };
    const IDENTITY = new THREE.Quaternion();
    const tmp = new THREE.Vector3();

    const video = videoRef.current;
    const overlay = overlayRef.current;
    const mount = mountRef.current;
    const octx = overlay.getContext("2d");

    // ---------- scene ----------
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.set(0, 0.3, 6.5);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setClearColor(0x000000, 1);
    renderer.setPixelRatio(window.devicePixelRatio);
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const key = new THREE.DirectionalLight(0xffffff, 2);
    key.position.set(3, 4, 5);
    const rim = new THREE.DirectionalLight(0xff2020, 2.5);
    rim.position.set(-4, -2, 2);
    scene.add(key, rim);

    // floor: dark red grid + ring
    const grid = new THREE.GridHelper(12, 24, 0x660000, 0x1f0000);
    grid.position.y = -1.9;
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(2.2, 2.23, 96),
      new THREE.MeshBasicMaterial({ color: 0xe10600, side: THREE.DoubleSide })
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = -1.9;
    scene.add(grid, ring);

    const holo = new THREE.Group();
    scene.add(holo);

    // shared materials
    const fillMat = new THREE.MeshBasicMaterial({
      color: 0x000000,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
      side: THREE.DoubleSide,
    });
    const solidMat = new THREE.MeshStandardMaterial({
      color: 0xd90000,
      roughness: 0.4,
      metalness: 0.25,
      side: THREE.DoubleSide,
    });
    const edgeMat = new THREE.LineBasicMaterial({ color: 0xffffff });
    const hiMat = new THREE.LineBasicMaterial({ color: 0xe10600, depthTest: false });

    const target = { rx: 0.35, ry: 0.4, s: 1 };
    const prev = { p: null, d: null };

    // ---------- model helpers ----------
    const prepare = (root, edgeAngle) => {
      root.traverse((o) => {
        if (o.isMesh) {
          o.userData.orig = o.material;
          const edges = new THREE.LineSegments(
            new THREE.EdgesGeometry(o.geometry, edgeAngle),
            edgeMat
          );
          edges.visible = false;
          o.add(edges);
          o.userData.edges = edges;
        }
      });
    };

    const applyMode = (root, m) => {
      root.traverse((o) => {
        if (!o.isMesh) return;
        o.userData.edges.visible = m === "wire" || (selected && selected.mesh === o);
        if (m === "wire") o.material = fillMat;
        else if (m === "solid") o.material = solidMat;
        else o.material = o.userData.orig;
      });
    };

    const fit = (obj, size = 2.6) => {
      const box = new THREE.Box3().setFromObject(obj);
      const center = box.getCenter(new THREE.Vector3());
      const dim = box.getSize(new THREE.Vector3());
      const max = Math.max(dim.x, dim.y, dim.z) || 1;
      obj.position.sub(center);
      const wrapper = new THREE.Group();
      wrapper.add(obj);
      wrapper.scale.setScalar(size / max);
      return wrapper;
    };

    // ---------- parts / selection ----------
    // flatten every mesh into the wrapper so each one can move independently
    const buildParts = (wrapped) => {
      const meshes = [];
      wrapped.traverse((o) => o.isMesh && meshes.push(o));
      if (meshes.length < 2) return [];
      wrapped.updateMatrixWorld(true);
      meshes.forEach((m) => wrapped.attach(m)); // attach keeps the world transform
      wrapped.updateMatrixWorld(true);
      const span = 2.6 / wrapped.scale.x; // model size in wrapper units
      return meshes.map((m, i) => {
        const centerWorld = new THREE.Box3().setFromObject(m).getCenter(new THREE.Vector3());
        const c = wrapped.worldToLocal(centerWorld); // part centre, wrapper space
        const dir = c.clone();
        if (dir.length() < 1e-6) dir.set(Math.sin(i * 12.9), Math.cos(i * 7.7), Math.sin(i * 3.1));
        dir.normalize();
        return {
          name: m.name || "Part " + (i + 1),
          mesh: m,
          home: m.position.clone(), // assembled position
          q0: m.quaternion.clone(), // assembled rotation
          cOff: c.clone().sub(m.position), // mesh origin -> part centre (spin pivot)
          vec: c.clone().multiplyScalar(EXPLODE_SPREAD).addScaledVector(dir, span * EXPLODE_PUSH),
          offset: new THREE.Vector3(), // user move
          rot: new THREE.Quaternion(), // user spin
        };
      });
    };

    const select = (p) => {
      if (selected === p) return;
      if (selected) {
        const e = selected.mesh.userData.edges;
        e.material = edgeMat;
        e.renderOrder = 0;
        e.visible = viewMode === "wire";
      }
      selected = p;
      if (p) {
        const e = p.mesh.userData.edges;
        e.material = hiMat;
        e.renderOrder = 999;
        e.visible = true;
      }
      setSelName(p ? p.name : "");
    };

    const canGrab = () => ex.t === 1 && ex.v > 0.5;

    const pickAt = (ndc) => {
      if (!parts.length) return null;
      ray.setFromCamera(ndc, camera);
      const hits = ray.intersectObjects(parts.map((p) => p.mesh), false);
      if (hits.length) return parts.find((p) => p.mesh === hits[0].object);
      return null;
    };

    // world-space direction -> wrapper-local direction
    const toLocalDelta = (d) => {
      const l0 = current.worldToLocal(new THREE.Vector3());
      return current.worldToLocal(d.clone()).sub(l0);
    };
    const camAxis = (i) => new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, i);

    // move in the plane facing the screen (dx, dy = fraction of the screen)
    const movePart = (p, dx, dy, gain) => {
      const wp = p.mesh.getWorldPosition(new THREE.Vector3());
      const vh = 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * camera.position.distanceTo(wp);
      const vw = vh * camera.aspect;
      const d = camAxis(0).multiplyScalar(dx * vw * gain).add(camAxis(1).multiplyScalar(-dy * vh * gain));
      p.offset.add(toLocalDelta(d));
    };

    // push toward / away from the camera
    const pushPart = (p, amt) => p.offset.add(toLocalDelta(camAxis(2).multiplyScalar(amt)));

    // spin the part around its own centre (dx, dy in pixels)
    const spinPart = (p, dx, dy) => {
      const inv = current.getWorldQuaternion(new THREE.Quaternion()).invert();
      const qy = new THREE.Quaternion().setFromAxisAngle(camAxis(1).applyQuaternion(inv), dx * 0.01);
      const qx = new THREE.Quaternion().setFromAxisAngle(camAxis(0).applyQuaternion(inv), dy * 0.01);
      p.rot.premultiply(qy).premultiply(qx);
    };

    const disposeCurrent = () => {
      select(null);
      drag = null;
      parts = [];
      ex.t = ex.v = 0;
      if (!current) return;
      holo.remove(current);
      current.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
      });
      current = null;
    };

    const place = (obj, edgeAngle) => {
      disposeCurrent();
      const wrapped = fit(obj);
      holo.add(wrapped);
      current = wrapped;
      parts = buildParts(wrapped);
      prepare(wrapped, edgeAngle);
      applyMode(wrapped, viewMode);
      target.s = 1;
    };

    const showPreset = (name) => {
      const g = new THREE.Group();
      g.add(new THREE.Mesh(PRESETS[name](), solidMat));
      place(g, 1);
    };

    const loadFile = (file) =>
      new Promise((resolve, reject) => {
        const ext = file.name.split(".").pop().toLowerCase();
        const url = URL.createObjectURL(file);
        const done = (obj) => {
          URL.revokeObjectURL(url);
          place(obj, 25);
          resolve(parts.length);
        };
        const fail = (e) => {
          URL.revokeObjectURL(url);
          reject(e);
        };
        if (ext === "glb" || ext === "gltf") {
          new GLTFLoader().load(url, (g) => done(g.scene), undefined, fail);
        } else if (ext === "obj") {
          new OBJLoader().load(url, done, undefined, fail);
        } else if (ext === "stl") {
          new STLLoader().load(
            url,
            (geo) => {
              geo.computeVertexNormals();
              done(new THREE.Mesh(geo, solidMat));
            },
            undefined,
            fail
          );
        } else {
          URL.revokeObjectURL(url);
          reject(new Error("Unsupported file type: ." + ext));
        }
      });

    api.current = {
      showPreset,
      loadFile,
      deselect: () => select(null),
      setMode: (m) => {
        viewMode = m;
        if (current) applyMode(current, m);
      },
      setExplode: (on) => {
        select(null);
        drag = null;
        ex.t = on ? 1 : 0;
        if (on) target.s = Math.min(target.s, 0.6); // zoom out so the spread fits
      },
      reset: () => {
        parts.forEach((p) => {
          p.offset.set(0, 0, 0);
          p.rot.identity();
        });
        target.rx = 0.35;
        target.ry = 0.4;
        target.s = ex.t ? 0.6 : 1;
      },
    };
    showPreset("Pyramid");

    // ---------- sizing ----------
    const resize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resize();
    window.addEventListener("resize", resize);
    overlay.width = 320;
    overlay.height = 240;

    // ---------- gestures ----------
    const setLabel = (t) => {
      if (t !== lastLabel) {
        lastLabel = t;
        setStatus(t);
      }
    };

    const drawHands = (hands, pinching) => {
      const w = overlay.width;
      const h = overlay.height;
      octx.clearRect(0, 0, w, h);
      hands.forEach((l, i) => {
        octx.strokeStyle = "rgba(255,255,255,0.85)";
        octx.lineWidth = 2;
        CONNECTIONS.forEach(([a, b]) => {
          octx.beginPath();
          octx.moveTo((1 - l[a].x) * w, l[a].y * h);
          octx.lineTo((1 - l[b].x) * w, l[b].y * h);
          octx.stroke();
        });
        octx.fillStyle = "#e10600";
        l.forEach((p) => {
          octx.beginPath();
          octx.arc((1 - p.x) * w, p.y * h, 3, 0, Math.PI * 2);
          octx.fill();
        });
        if (pinching[i]) {
          octx.strokeStyle = "#e10600";
          octx.lineWidth = 3;
          octx.beginPath();
          octx.arc(
            (1 - (l[4].x + l[8].x) / 2) * w,
            ((l[4].y + l[8].y) / 2) * h,
            14,
            0,
            Math.PI * 2
          );
          octx.stroke();
        }
      });
    };

    const handleGestures = (hands) => {
      const info = hands.map((l) => ({
        pinch: dist(l[4], l[8]) < PINCH_THRESHOLD,
        fist: isFist(l),
        x: 1 - (l[4].x + l[8].x) / 2,
        y: (l[4].y + l[8].y) / 2,
      }));
      drawHands(hands, info.map((i) => i.pinch));

      // a part picked with the MOUSE owns the hand gestures; otherwise they act on the whole model
      const sel = selected && canGrab() ? selected : null;

      if (info.length === 2 && info[0].pinch && info[1].pinch) {
        const d = Math.hypot(info[0].x - info[1].x, info[0].y - info[1].y);
        if (prev.d) {
          if (sel) pushPart(sel, (d - prev.d) * 12);
          else target.s = THREE.MathUtils.clamp(target.s * (d / prev.d), 0.3, 3);
        }
        prev.d = d;
        prev.p = null;
        setLabel(sel ? "PUSH / PULL PART" : "ZOOMING");
      } else if (info.some((i) => i.pinch)) {
        const p = info.find((i) => i.pinch);
        if (prev.p) {
          if (sel) {
            if (!drag) movePart(sel, p.x - prev.p.x, p.y - prev.p.y, 1.4);
          } else {
            target.ry += (p.x - prev.p.x) * 6;
            target.rx += (p.y - prev.p.y) * 6;
          }
        }
        prev.p = { x: p.x, y: p.y };
        prev.d = null;
        setLabel(sel ? "MOVING PART" : "ROTATING");
      } else if (info.some((i) => i.fist)) {
        if (sel) {
          sel.offset.set(0, 0, 0);
          sel.rot.identity();
          setLabel("PART RESET");
        } else {
          api.current.reset();
          setLabel("RESET");
        }
        prev.p = prev.d = null;
      } else {
        prev.p = prev.d = null;
        setLabel(info.length ? "HAND FOUND - PINCH TO GRAB" : "SHOW YOUR HAND");
      }
    };

    // ---------- mouse ----------
    const dom = renderer.domElement;
    const ndcOf = (e) => {
      const r = dom.getBoundingClientRect();
      return new THREE.Vector2(
        ((e.clientX - r.left) / r.width) * 2 - 1,
        -((e.clientY - r.top) / r.height) * 2 + 1
      );
    };
    const onDown = (e) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
      mouse.dist = 0;
      if (e.button === 2) {
        mouse.mode = selected && canGrab() ? "spin" : null;
      } else if (e.button === 0) {
        const hit = canGrab() ? pickAt(ndcOf(e)) : null;
        if (hit) {
          select(hit);
          drag = { part: hit, src: "mouse" };
          mouse.mode = "move";
        } else {
          mouse.mode = "rotate";
        }
      } else return;
      dom.setPointerCapture(e.pointerId);
      dom.style.cursor = "grabbing";
    };
    const onMove = (e) => {
      if (!mouse.mode) {
        dom.style.cursor = canGrab() && pickAt(ndcOf(e)) ? "pointer" : "grab";
        return;
      }
      const dx = e.clientX - mouse.x;
      const dy = e.clientY - mouse.y;
      mouse.x = e.clientX;
      mouse.y = e.clientY;
      mouse.dist += Math.abs(dx) + Math.abs(dy);
      if (mouse.mode === "move" && drag) {
        const r = dom.getBoundingClientRect();
        movePart(drag.part, dx / r.width, dy / r.height, 1);
      } else if (mouse.mode === "spin" && selected) {
        spinPart(selected, dx, dy);
      } else if (mouse.mode === "rotate") {
        target.ry += dx * 0.008;
        target.rx += dy * 0.008;
      }
    };
    const onUp = () => {
      if (!mouse.mode) return;
      if (drag && drag.src === "mouse") drag = null;
      mouse.mode = null;
      dom.style.cursor = "grab";
    };
    const onWheel = (e) => {
      e.preventDefault();
      const d = e.deltaY || e.deltaX; // shift+wheel can arrive as deltaX
      if (e.shiftKey && selected && canGrab()) {
        pushPart(selected, -d * 0.004);
      } else {
        target.s = THREE.MathUtils.clamp(target.s * Math.exp(-d * 0.001), 0.3, 3);
      }
    };
    const onKey = (e) => {
      if (e.key === "Escape") select(null);
    };
    const noMenu = (e) => e.preventDefault();
    dom.style.cursor = "grab";
    dom.style.touchAction = "none";
    dom.addEventListener("pointerdown", onDown);
    dom.addEventListener("pointermove", onMove);
    dom.addEventListener("pointerup", onUp);
    dom.addEventListener("pointercancel", onUp);
    dom.addEventListener("wheel", onWheel, { passive: false });
    dom.addEventListener("contextmenu", noMenu);
    window.addEventListener("keydown", onKey);

    // ---------- loop ----------
    const loop = () => {
      raf = requestAnimationFrame(loop);
      let seen = false;
      if (landmarker && video.readyState >= 2 && video.currentTime !== lastTime) {
        lastTime = video.currentTime;
        const res = landmarker.detectForVideo(video, performance.now());
        seen = res.landmarks.length > 0;
        handleGestures(res.landmarks);
        if (!seen) {
          octx.clearRect(0, 0, overlay.width, overlay.height);
          prev.p = prev.d = null;
        }
      }

      // explode animation + per-part transforms (spin happens around the part's own centre)
      ex.v += (ex.t - ex.v) * 0.08;
      parts.forEach((p) => {
        if (!ex.t) {
          p.offset.multiplyScalar(0.85);
          p.rot.slerp(IDENTITY, 0.15);
        }
        p.mesh.position
          .copy(p.home)
          .addScaledVector(p.vec, ex.v)
          .add(p.offset)
          .add(p.cOff)
          .sub(tmp.copy(p.cOff).applyQuaternion(p.rot));
        p.mesh.quaternion.copy(p.rot).multiply(p.q0);
      });

      holo.rotation.x += (target.rx - holo.rotation.x) * 0.15;
      holo.rotation.y += (target.ry - holo.rotation.y) * 0.15;
      holo.scale.setScalar(holo.scale.x + (target.s - holo.scale.x) * 0.15);
      ring.rotation.z += 0.004;
      renderer.render(scene, camera);
    };
    loop();

    // ---------- camera + tracker ----------
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
        });
        if (cancelled) return stream.getTracks().forEach((t) => t.stop());
        video.srcObject = stream;
        await video.play();
        const fileset = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        landmarker = await HandLandmarker.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
        });
        setStatus("SHOW YOUR HAND");
      } catch (err) {
        console.error(err);
        setStatus("CAMERA ERROR");
        setError(err.message);
      }
    })();

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      window.removeEventListener("keydown", onKey);
      dom.removeEventListener("pointerdown", onDown);
      dom.removeEventListener("pointermove", onMove);
      dom.removeEventListener("pointerup", onUp);
      dom.removeEventListener("pointercancel", onUp);
      dom.removeEventListener("wheel", onWheel);
      dom.removeEventListener("contextmenu", noMenu);
      stream?.getTracks().forEach((t) => t.stop());
      landmarker?.close();
      disposeCurrent();
      renderer.dispose();
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
    };
  }, []);

  // ---------- UI handlers ----------
  const pickPreset = (name) => {
    setError("");
    api.current.showPreset(name);
    setActive(name);
    setCustom(false);
    setPartCount(0);
    setExploded(false);
  };

  const toggleExplode = () => {
    const next = !exploded;
    setExploded(next);
    api.current.setExplode(next);
  };

  const changeMode = (m) => {
    setModeState(m);
    api.current.setMode(m);
  };

  const upload = (file) => {
    if (!file) return;
    setError("");
    api.current
      .loadFile(file)
      .then((n) => {
        setActive(file.name);
        setCustom(true);
        setPartCount(n);
        setExploded(false);
      })
      .catch((e) => setError(e.message || "Could not load that model"));
  };

  return (
    <div
      className="app"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        upload(e.dataTransfer.files?.[0]);
      }}
    >
      <div ref={mountRef} className="three" />

      <header className="topbar">
        <span className="brand">MACERENA WING<b>/</b>CR7 MV 3</span>
        <span className="status">{status}</span>
      </header>

      <aside className="panel">
        <h3>Models</h3>
        <div className="grid">
          {Object.keys(PRESETS).map((n) => (
            <button key={n} className={active === n ? "on" : ""} onClick={() => pickPreset(n)}>
              {n}
            </button>
          ))}
        </div>

        <h3>Style</h3>
        <div className="seg">
          {MODES.map(([k, label]) => (
            <button key={k} className={mode === k ? "on" : ""} onClick={() => changeMode(k)}>
              {label}
            </button>
          ))}
        </div>

        <h3>Your model</h3>
        <button className="upload" onClick={() => fileRef.current.click()}>
          + Upload .glb / .obj / .stl
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".glb,.gltf,.obj,.stl"
          hidden
          onChange={(e) => {
            upload(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
        <p className="hint">or drag a file onto the window</p>
        <p className="current">Showing: {active}</p>
        {error && <p className="error">{error}</p>}

        {custom && (
          <>
            <h3>Explode</h3>
            <button
              className={`explode ${exploded ? "on" : ""}`}
              disabled={partCount < 2}
              onClick={toggleExplode}
            >
              {exploded ? "Collapse view" : "Exploded view"}
            </button>
            <p className="hint">
              {partCount < 2
                ? "This model has only one part."
                : `${partCount} parts - click one to select it`}
            </p>
            {exploded && (
              <>
                <p className="current">Selected: {selName || "none"}</p>
                <button
                  className="explode deselect"
                  disabled={!selName}
                  onClick={() => api.current.deselect()}
                >
                  Deselect
                </button>
                <p className="hint">
                  Click a part with the mouse to select it (stays until you click another). Then your hand controls only that part: pinch + move, two-hand pinch = push/pull, fist = reset part
                </p>
              </>
            )}
          </>
        )}

        <button className="reset" onClick={() => api.current.reset()}>
          Reset view
        </button>
      </aside>

      <div className="cam">
        <video ref={videoRef} playsInline muted />
        <canvas ref={overlayRef} />
        <span className="tag">CAM</span>
      </div>

      <footer className="help">
        Pinch + move: rotate &nbsp;/&nbsp; Two-hand pinch: zoom &nbsp;/&nbsp; Fist: reset &nbsp;/&nbsp; Exploded: click a part, then hand controls it
      </footer>
    </div>
  );
}