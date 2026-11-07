'use strict';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const API = '';
const roomNames = {
  neural: 'NEURAL LABORATORY', scanner: 'MRI SCANNER CHAMBER', explore: 'TUMOUR EXPLORATION ROOM',
  compare: 'CASE COMPARISON LABORATORY', research: 'RESEARCH OBSERVATORY', atlas: 'NEURO-ONCOLOGY ATLAS'
};
const state = {
  room: 'neural', files: [], currentFile: null, result: null, explanation: null, caseId: null,
  annotatedUrl: '', heatmapUrl: '', viewerMode: 'original', history: [], selectedHistory: null,
  favoritesOnly: false, similarity: [], similarLoading: false, region: 'Frontal lobe', quizIndex: 0, brightness: 0,
  contrast: 100, gamma: 1, invert: false, grayscale: true, zoom: 1, panX: 0, panY: 0,
  image: null, annotatedImage: null, heatmapImage: null, viewerRect: null,
  measurePoints: [], roi: null, pointerStart: null, isPanning: false,
  compareBeforeAfter: false, compareSplit: 50,
  quizProgress: (() => { try { return JSON.parse(localStorage.getItem('tscannerQuizProgress') || '{"answered":0,"correct":0}'); } catch (_) { return { answered: 0, correct: 0 }; } })(),
  referenceFavorites: (() => { try { return JSON.parse(localStorage.getItem('tscannerReferenceFavorites') || '[]'); } catch (_) { return []; } })()
};

function toast(message, duration = 2800) {
  const el = $('#toast');
  el.textContent = message;
  el.classList.add('show');
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => el.classList.remove('show'), duration);
}

async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, { cache: 'no-store', ...options });
  const contentType = response.headers.get('content-type') || '';
  const payload = contentType.includes('application/json') ? await response.json() : await response.text();
  if (!response.ok) {
    const message = typeof payload === 'object' && payload ? (payload.detail || payload.message) : payload;
    throw new Error(message || `Request failed (${response.status})`);
  }
  return payload;
}

function setRoom(name) {
  if (!roomNames[name]) return;
  if (name !== state.room) state.previousRoom = state.room;
  state.room = name;
  $$('[data-room-panel]').forEach(panel => {
    const active = panel.dataset.roomPanel === name;
    panel.hidden = false;
    panel.classList.toggle('active', active);
    if (!active) panel.hidden = true;
  });
  $$('.env-button').forEach(button => button.classList.toggle('active', button.dataset.room === name));
  $('#environmentName').textContent = roomNames[name];
  if (name === 'research') loadResearch();
  if (name === 'compare') {
    $('#compareCategory').textContent = state.result?.predicted_type ? state.result.predicted_type.toUpperCase() : 'NO PREDICTION SELECTED';
    $('#runSimilar').disabled = !state.currentFile || !state.result;
  }
  if (name === 'explore') drawMRI();
  if (name === 'atlas' && !$('#atlasContent').innerHTML.trim()) renderAtlas('glioma');
}

$$('[data-room]').forEach(button => button.addEventListener('click', () => setRoom(button.dataset.room)));
$('#helpButton').addEventListener('click', () => $('#helpModal').classList.remove('hidden'));
$('#closeHelp').addEventListener('click', () => $('#helpModal').classList.add('hidden'));
window.addEventListener('keydown', event => {
  if (event.key === 'Escape') {
    $('#helpModal').classList.add('hidden');
    if (state.room !== 'neural') setRoom(state.previousRoom || 'neural');
  }
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLSelectElement) return;
  const routes = { '1': 'neural', '2': 'scanner', '3': 'explore', '4': 'compare', '5': 'research', '6': 'atlas' };
  if (routes[event.key]) setRoom(routes[event.key]);
  if (event.key.toLowerCase() === 'h') $('#helpModal').classList.toggle('hidden');
});

async function refreshBackend() {
  try {
    const [health, model] = await Promise.all([api('/api/health'), api('/api/model')]);
    const status = $('#backendStatus');
    status.classList.toggle('online', health.backend === 'online');
    $('span:last-child', status).textContent = health.backend === 'online' ? 'BACKEND ONLINE' : 'BACKEND OFFLINE';
    $('#researchModelState').textContent = model.loaded ? 'READY' : 'OFFLINE';
    const threshold = Math.round(Number(model.confidence_threshold ?? .25) * 100);
    $('#thresholdRange').value = threshold; $('#thresholdValue').textContent = `${threshold}%`;
    $('#researchDevice').textContent = `DEVICE ${String(model.device || 'UNKNOWN').toUpperCase()}`;
    $('#runtimeModel').textContent = `MODEL: ${model.loaded ? 'LOADED' : 'NOT LOADED'}`;
    $('#runtimeBackend').textContent = `BACKEND: ${health.backend || 'UNKNOWN'}`;
    const classNames = Object.values(model.classes || {}).join(', ');
    $('#runtimeClasses').textContent = `CLASSES: ${classNames || 'NOT AVAILABLE'}`;
    return { health, model };
  } catch (error) {
    const status = $('#backendStatus');
    status.classList.remove('online');
    $('span:last-child', status).textContent = 'BACKEND OFFLINE';
    $('#researchModelState').textContent = 'OFFLINE';
    $('#researchDevice').textContent = 'DEVICE UNKNOWN';
    $('#runtimeBackend').textContent = 'BACKEND: OFFLINE';
    return null;
  }
}
refreshBackend();

// ---------------------------------------------------------------------------
// Native glTF 2.0 / GLB brain renderer. Vanilla JavaScript + WebGL 1 only.
// Loads the user's model and texture files; no Three.js or other JS libraries.
// ---------------------------------------------------------------------------
const BrainRenderer = (() => {
  const canvas = $('#brainCanvas');
  const stage = $('#brainStage');
  const status = $('#brainLoadStatus');
  const MODEL_URL = '/assets/brain/source/brain.glb';
  const TEXTURE_URLS = {
    baseColor: '/assets/brain/textures/gltf_embedded_0.png',
    metallic: '/assets/brain/textures/gltf_embedded_1.png',
    roughness: '/assets/brain/textures/gltf_embedded_1@channels=A.png',
    normal: '/assets/brain/textures/gltf_embedded_2.png'
  };
  const setStatus = (message, stateName = '') => {
    if (!status) return;
    status.textContent = message;
    status.dataset.state = stateName;
    status.classList.toggle('hidden', !message);
  };

  if (!canvas || !window.WebGLRenderingContext) {
    setStatus('WEBGL NOT AVAILABLE · ENABLE HARDWARE ACCELERATION', 'error');
    return null;
  }
  const gl = canvas.getContext('webgl', { alpha: true, antialias: true, premultipliedAlpha: false }) || canvas.getContext('experimental-webgl');
  if (!gl) {
    setStatus('WEBGL NOT AVAILABLE · BRAIN MODEL CANNOT BE RENDERED', 'error');
    return null;
  }

  const vertexSource = `
    attribute vec3 aPosition;
    attribute vec3 aNormal;
    attribute vec2 aUV;
    attribute vec4 aTangent;
    uniform mat4 uMVP;
    uniform mat4 uModel;
    varying vec3 vNormal;
    varying vec3 vWorld;
    varying vec2 vUV;
    varying vec4 vTangent;
    void main(){
      vec4 world = uModel * vec4(aPosition, 1.0);
      vWorld = world.xyz;
      vNormal = normalize(mat3(uModel) * aNormal);
      vUV = aUV;
      vTangent = vec4(normalize(mat3(uModel) * aTangent.xyz), aTangent.w);
      gl_Position = uMVP * vec4(aPosition, 1.0);
    }
  `;
  const fragmentSource = `
    precision highp float;
    uniform vec4 uBaseColor;
    uniform sampler2D uBaseMap;
    uniform sampler2D uMetalRoughMap;
    uniform sampler2D uMetalMap;
    uniform sampler2D uRoughMap;
    uniform sampler2D uNormalMap;
    uniform float uHasBaseMap;
    uniform float uHasMetalRoughMap;
    uniform float uHasMetalMap;
    uniform float uHasRoughMap;
    uniform float uHasNormalMap;
    uniform float uMetallic;
    uniform float uRoughness;
    uniform float uOpacity;
    uniform float uCutaway;
    uniform float uAlphaCutoff;
    uniform float uMaterialView;
    varying vec3 vNormal;
    varying vec3 vWorld;
    varying vec2 vUV;
    varying vec4 vTangent;
    void main(){
      if(uCutaway > 0.5 && vWorld.x > 0.0) discard;
      vec4 base = uBaseColor;
      if(uHasBaseMap > 0.5) base *= texture2D(uBaseMap, vUV);
      if(base.a < uAlphaCutoff) discard;
      float metallic = uMetallic;
      float roughness = uRoughness;
      if(uHasMetalRoughMap > 0.5){
        vec4 mr = texture2D(uMetalRoughMap, vUV);
        if(uHasRoughMap < 0.5) roughness *= mr.g;
        if(uHasMetalMap < 0.5) metallic *= mr.b;
      }
      if(uHasMetalMap > 0.5) metallic = texture2D(uMetalMap, vUV).r;
      if(uHasRoughMap > 0.5) roughness = texture2D(uRoughMap, vUV).r;
      roughness = clamp(roughness, 0.08, 1.0);
      metallic = clamp(metallic, 0.0, 1.0);
      vec3 N = normalize(vNormal);
      vec3 normalSample = uHasNormalMap > 0.5 ? texture2D(uNormalMap, vUV).xyz : vec3(0.5, 0.5, 1.0);
      if(uHasNormalMap > 0.5){
        vec3 T = normalize(vTangent.xyz - dot(vTangent.xyz, N) * N);
        vec3 B = normalize(cross(N, T)) * vTangent.w;
        vec3 mapN = normalSample * 2.0 - 1.0;
        N = normalize(mat3(T, B, N) * mapN);
      }
      if(uMaterialView > 0.5){
        vec3 inspection = base.rgb;
        if(uMaterialView > 1.5 && uMaterialView < 2.5) inspection = vec3(metallic);
        else if(uMaterialView > 2.5 && uMaterialView < 3.5) inspection = vec3(roughness);
        else if(uMaterialView > 3.5) inspection = normalSample;
        gl_FragColor = vec4(inspection, base.a * uOpacity);
        return;
      }
      vec3 L = normalize(vec3(-0.48, 0.78, 0.86));
      vec3 V = normalize(vec3(0.0, 0.1, 3.6) - vWorld);
      vec3 H = normalize(L + V);
      float ndl = max(dot(N, L), 0.0);
      float ndv = max(dot(N, V), 0.0);
      float specPower = mix(96.0, 5.0, roughness * roughness);
      float specular = pow(max(dot(N, H), 0.0), specPower) * mix(0.16, 0.9, metallic);
      float fresnel = pow(1.0 - ndv, 3.0);
      vec3 diffuse = base.rgb * (0.25 + 0.78 * ndl) * (1.0 - metallic * 0.42);
      vec3 color = diffuse + vec3(0.82, 0.91, 1.0) * specular;
      color += vec3(0.12, 0.54, 0.62) * fresnel * 0.17;
      color = pow(max(color, vec3(0.0)), vec3(0.92));
      gl_FragColor = vec4(color, base.a * uOpacity);
    }
  `;

  function compileShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const message = gl.getShaderInfoLog(shader) || 'Unknown shader compilation error';
      gl.deleteShader(shader);
      throw new Error(message);
    }
    return shader;
  }
  const program = gl.createProgram();
  try {
    gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vertexSource));
    gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fragmentSource));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) || 'Shader link failed');
  } catch (error) {
    console.error('T-Scanner brain renderer:', error);
    setStatus('BRAIN RENDERER INITIALIZATION FAILED', 'error');
    return null;
  }
  gl.useProgram(program);
  const loc = {};
  [
    'aPosition', 'aNormal', 'aUV', 'aTangent', 'uMVP', 'uModel', 'uBaseColor', 'uBaseMap',
    'uMetalRoughMap', 'uMetalMap', 'uRoughMap', 'uNormalMap', 'uHasBaseMap', 'uHasMetalRoughMap',
    'uHasMetalMap', 'uHasRoughMap', 'uHasNormalMap', 'uMetallic', 'uRoughness', 'uOpacity', 'uCutaway', 'uAlphaCutoff', 'uMaterialView'
  ].forEach(name => { loc[name] = gl.getAttribLocation(program, name) >= 0 ? gl.getAttribLocation(program, name) : gl.getUniformLocation(program, name); });

  const identity = () => new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
  function multiply(a, b) {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      out[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
    return out;
  }
  const translate = (x, y, z) => { const m = identity(); m[12] = x; m[13] = y; m[14] = z; return m; };
  const scale = (x, y, z) => { const m = identity(); m[0] = x; m[5] = y; m[10] = z; return m; };
  const rotateX = a => { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]); };
  const rotateY = a => { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]); };
  function perspective(fov, aspect, near, far) {
    const f = 1 / Math.tan(fov / 2), nf = 1 / (near - far);
    return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
  }
  function composeNode(node) {
    if (node.matrix) return new Float32Array(node.matrix);
    const t = node.translation || [0, 0, 0], q = node.rotation || [0, 0, 0, 1], s = node.scale || [1, 1, 1];
    const [x, y, z, w] = q;
    const out = new Float32Array([
      (1-2*y*y-2*z*z)*s[0], (2*x*y+2*w*z)*s[0], (2*x*z-2*w*y)*s[0], 0,
      (2*x*y-2*w*z)*s[1], (1-2*x*x-2*z*z)*s[1], (2*y*z+2*w*x)*s[1], 0,
      (2*x*z+2*w*y)*s[2], (2*y*z-2*w*x)*s[2], (1-2*x*x-2*y*y)*s[2], 0,
      t[0], t[1], t[2], 1
    ]);
    return out;
  }
  function transformPoint(m, p) {
    const x=p[0], y=p[1], z=p[2];
    return [m[0]*x+m[4]*y+m[8]*z+m[12], m[1]*x+m[5]*y+m[9]*z+m[13], m[2]*x+m[6]*y+m[10]*z+m[14]];
  }

  const componentInfo = {
    5120: { bytes: 1, read: (v, o) => v.getInt8(o), signed: true, max: 127 },
    5121: { bytes: 1, read: (v, o) => v.getUint8(o), max: 255 },
    5122: { bytes: 2, read: (v, o) => v.getInt16(o, true), signed: true, max: 32767 },
    5123: { bytes: 2, read: (v, o) => v.getUint16(o, true), max: 65535 },
    5125: { bytes: 4, read: (v, o) => v.getUint32(o, true), max: 4294967295 },
    5126: { bytes: 4, read: (v, o) => v.getFloat32(o, true) }
  };
  const typeSize = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 };
  function readAccessor(gltf, binary, accessorIndex) {
    const accessor = gltf.accessors[accessorIndex];
    if (!accessor || accessor.sparse) throw new Error('Unsupported or missing glTF accessor');
    const info = componentInfo[accessor.componentType];
    if (!info) throw new Error('Unsupported glTF component type: ' + accessor.componentType);
    const count = accessor.count, n = typeSize[accessor.type];
    const result = new Float32Array(count * n);
    if (accessor.bufferView === undefined) return result;
    const viewDef = gltf.bufferViews[accessor.bufferView];
    const view = new DataView(binary, (viewDef.byteOffset || 0) + (accessor.byteOffset || 0), viewDef.byteLength - (accessor.byteOffset || 0));
    const stride = viewDef.byteStride || info.bytes * n;
    for (let i = 0; i < count; i++) for (let c = 0; c < n; c++) {
      let value = info.read(view, i * stride + c * info.bytes);
      if (accessor.normalized && accessor.componentType !== 5126) {
        value = info.signed ? Math.max(value / info.max, -1) : value / info.max;
      }
      result[i * n + c] = value;
    }
    return result;
  }
  function readIndices(gltf, binary, accessorIndex) {
    const accessor = gltf.accessors[accessorIndex], info = componentInfo[accessor.componentType];
    if (!accessor || !info || accessor.type !== 'SCALAR') throw new Error('Invalid glTF index accessor');
    const viewDef = gltf.bufferViews[accessor.bufferView];
    const view = new DataView(binary, (viewDef.byteOffset || 0) + (accessor.byteOffset || 0), viewDef.byteLength - (accessor.byteOffset || 0));
    const stride = viewDef.byteStride || info.bytes;
    const values = new (accessor.componentType === 5125 ? Uint32Array : Uint16Array)(accessor.count);
    for (let i = 0; i < accessor.count; i++) values[i] = info.read(view, i * stride);
    return values;
  }
  function makeNormals(positions, indices) {
    const normals = new Float32Array(positions.length);
    for (let i = 0; i < indices.length; i += 3) {
      const a=indices[i]*3, b=indices[i+1]*3, c=indices[i+2]*3;
      const ab=[positions[b]-positions[a],positions[b+1]-positions[a+1],positions[b+2]-positions[a+2]];
      const ac=[positions[c]-positions[a],positions[c+1]-positions[a+1],positions[c+2]-positions[a+2]];
      const nx=ab[1]*ac[2]-ab[2]*ac[1], ny=ab[2]*ac[0]-ab[0]*ac[2], nz=ab[0]*ac[1]-ab[1]*ac[0];
      [a,b,c].forEach(o => { normals[o]+=nx; normals[o+1]+=ny; normals[o+2]+=nz; });
    }
    for (let i=0;i<normals.length;i+=3) { const n=Math.hypot(normals[i],normals[i+1],normals[i+2])||1; normals[i]/=n; normals[i+1]/=n; normals[i+2]/=n; }
    return normals;
  }
  function makeTangents(positions, normals, uvs, indices) {
    const tan1 = new Float32Array(positions.length), tan2 = new Float32Array(positions.length);
    for (let i=0;i<indices.length;i+=3) {
      const i1=indices[i], i2=indices[i+1], i3=indices[i+2];
      const p1=i1*3,p2=i2*3,p3=i3*3,t1=i1*2,t2=i2*2,t3=i3*2;
      const x1=positions[p2]-positions[p1], x2=positions[p3]-positions[p1], y1=positions[p2+1]-positions[p1+1], y2=positions[p3+1]-positions[p1+1], z1=positions[p2+2]-positions[p1+2], z2=positions[p3+2]-positions[p1+2];
      const s1=uvs[t2]-uvs[t1], s2=uvs[t3]-uvs[t1], r1=uvs[t2+1]-uvs[t1+1], r2=uvs[t3+1]-uvs[t1+1];
      const denom=s1*r2-s2*r1; if(Math.abs(denom)<1e-9) continue;
      const r=1/denom, sx=(r2*x1-r1*x2)*r, sy=(r2*y1-r1*y2)*r, sz=(r2*z1-r1*z2)*r, tx=(s1*x2-s2*x1)*r, ty=(s1*y2-s2*y1)*r, tz=(s1*z2-s2*z1)*r;
      [i1,i2,i3].forEach(v=>{const o=v*3;tan1[o]+=sx;tan1[o+1]+=sy;tan1[o+2]+=sz;tan2[o]+=tx;tan2[o+1]+=ty;tan2[o+2]+=tz;});
    }
    const out=new Float32Array(normals.length/3*4);
    for(let i=0;i<normals.length/3;i++){
      const o=i*3,d=i*4,nx=normals[o],ny=normals[o+1],nz=normals[o+2];
      let tx=tan1[o],ty=tan1[o+1],tz=tan1[o+2],dot=nx*tx+ny*ty+nz*tz;
      tx-=nx*dot;ty-=ny*dot;tz-=nz*dot;const length=Math.hypot(tx,ty,tz)||1;tx/=length;ty/=length;tz/=length;
      const cx=ny*tz-nz*ty,cy=nz*tx-nx*tz,cz=nx*ty-ny*tx;
      const handed=cx*tan2[o]+cy*tan2[o+1]+cz*tan2[o+2]<0?-1:1;
      out[d]=tx;out[d+1]=ty;out[d+2]=tz;out[d+3]=handed;
    }
    return out;
  }
  function bufferData(target, data) {
    const buffer=gl.createBuffer();gl.bindBuffer(target,buffer);gl.bufferData(target,data,gl.STATIC_DRAW);return buffer;
  }
  function createGLTexture(image) {
    const texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,image);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.bindTexture(gl.TEXTURE_2D,null);
    return texture;
  }
  function solidTexture(r,g,b,a=255) {
    const texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array([r,g,b,a]));
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    return texture;
  }
  const defaults = { white: solidTexture(255,255,255), normal: solidTexture(128,128,255), black: solidTexture(0,0,0) };
  function loadImage(url) {
    return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(createGLTexture(image));image.onerror=()=>reject(new Error('Could not load texture: '+url));image.src=url;});
  }
  async function parseGLB(arrayBuffer) {
    const header=new DataView(arrayBuffer);
    if(header.byteLength<20||header.getUint32(0,true)!==0x46546c67) throw new Error('The brain model is not a valid GLB file');
    if(header.getUint32(4,true)!==2) throw new Error('Only glTF 2.0 GLB models are supported');
    let offset=12,json=null,binary=null;
    while(offset+8<=header.byteLength){
      const length=header.getUint32(offset,true),type=header.getUint32(offset+4,true);offset+=8;
      if(offset+length>header.byteLength) throw new Error('The GLB file contains an invalid chunk');
      if(type===0x4e4f534a) json=JSON.parse(new TextDecoder().decode(new Uint8Array(arrayBuffer,offset,length)).replace(/\u0000+$/g,''));
      else if(type===0x004e4942) binary=arrayBuffer.slice(offset,offset+length);
      offset+=length;
    }
    if(!json) throw new Error('No glTF scene data was found in brain.glb');
    if(json.buffers?.[0]?.uri){const response=await fetch(new URL(json.buffers[0].uri,location.origin+MODEL_URL));if(!response.ok)throw new Error('The GLB external geometry buffer could not be loaded');binary=await response.arrayBuffer();}
    if(!binary) binary=new ArrayBuffer(0);
    return {json,binary};
  }
  async function loadGLTFImage(gltf,binary,imageIndex) {
    const image=gltf.images?.[imageIndex];if(!image) return null;
    if(image.uri){
      const url=image.uri.startsWith('data:')?image.uri:new URL(image.uri,location.origin+MODEL_URL).href;
      return loadImage(url);
    }
    if(image.bufferView!==undefined){
      const view=gltf.bufferViews[image.bufferView],blob=new Blob([binary.slice(view.byteOffset||0,(view.byteOffset||0)+view.byteLength)],{type:image.mimeType||'image/png'});
      const objectURL=URL.createObjectURL(blob);
      try{return await loadImage(objectURL);}finally{URL.revokeObjectURL(objectURL);}
    }
    return null;
  }
  async function loadTextureIndex(gltf,binary,textureIndex,cache) {
    if(textureIndex===undefined||textureIndex===null)return null;
    if(cache.has(textureIndex))return cache.get(textureIndex);
    const textureDef=gltf.textures?.[textureIndex];if(!textureDef)return null;
    const promise=loadGLTFImage(gltf,binary,textureDef.source);cache.set(textureIndex,promise);return promise;
  }
  function classifyPart(name) {
    const value=(name||'').toLowerCase();
    if(/cerebell|vermis/.test(value))return 'cerebellum';
    if(/brain.?stem|medulla|pons/.test(value))return 'stem';
    if(/tumou?r|lesion/.test(value))return 'tumor';
    return 'cerebrum';
  }

  let parts=[], modelCenter=[0,0,0], fitScale=1, loaded=false, running=true;
  let rotationX=-.12,rotationY=-.45,targetX=rotationX,targetY=rotationY,zoom=1,dragging=false,lastX=0,lastY=0,pointerY=0;
  let exploded=false,cutaway=false,transparent=false,selectedRegion=null,tumorHighlight=false;
  let materialView = 0;
  const materialFilterValues = { main: 0, base: 1, metal: 2, rough: 3, normal: 4 };
  const layerState={cerebrum:true,cerebellum:true,stem:true,tumor:true};
  async function loadModel(){
    setStatus('LOADING BRAIN ANATOMY MODEL…','loading');
    const response=await fetch(MODEL_URL,{cache:'no-store'});
    if(!response.ok)throw new Error('brain.glb was not found at backend/3d/brain/source/brain.glb');
    const {json,binary}=await parseGLB(await response.arrayBuffer());
    if(!json.meshes?.length)throw new Error('The GLB file does not contain any mesh geometry');
    const textureCache=new Map();
    const [overrideBase,overrideMetal,overrideRough,overrideNormal]=await Promise.all(
      Object.values(TEXTURE_URLS).map(url=>loadImage(url).catch(()=>null))
    );
    const overrides={base:overrideBase,metal:overrideMetal,rough:overrideRough,normal:overrideNormal};
    let min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
    const scene=json.scenes?.[json.scene||0];
    const nodes=json.nodes||[];
    const roots=scene?.nodes||nodes.map((node,index)=>index).filter(index=>!nodes.some(parent=>(parent.children||[]).includes(index)));
    async function visit(nodeIndex,parentMatrix){
      const node=nodes[nodeIndex];if(!node)return;
      const nodeMatrix=multiply(parentMatrix,composeNode(node));
      if(node.mesh!==undefined){
        const mesh=json.meshes[node.mesh];
        for (const primitive of (mesh.primitives||[])) {
          if((primitive.mode??4)!==4||primitive.attributes?.POSITION===undefined)continue;
          const positions=readAccessor(json,binary,primitive.attributes.POSITION);
          const normals=primitive.attributes.NORMAL!==undefined?readAccessor(json,binary,primitive.attributes.NORMAL):null;
          const uvs=primitive.attributes.TEXCOORD_0!==undefined?readAccessor(json,binary,primitive.attributes.TEXCOORD_0):new Float32Array(positions.length/3*2);
          let indices=primitive.indices!==undefined?readIndices(json,binary,primitive.indices):(positions.length/3>65535?Uint32Array.from({length:positions.length/3},(_,i)=>i):Uint16Array.from({length:positions.length/3},(_,i)=>i));
          const normalData=normals||makeNormals(positions,indices);
          const tangentData=primitive.attributes.TANGENT!==undefined?readAccessor(json,binary,primitive.attributes.TANGENT):makeTangents(positions,normalData,uvs,indices);
          const pBuffer=bufferData(gl.ARRAY_BUFFER,positions),nBuffer=bufferData(gl.ARRAY_BUFFER,normalData),uvBuffer=bufferData(gl.ARRAY_BUFFER,uvs),tBuffer=bufferData(gl.ARRAY_BUFFER,tangentData),iBuffer=bufferData(gl.ELEMENT_ARRAY_BUFFER,indices);
          const indexType=indices instanceof Uint32Array?gl.UNSIGNED_INT:gl.UNSIGNED_SHORT;
          if(indexType===gl.UNSIGNED_INT&&!gl.getExtension('OES_element_index_uint'))throw new Error('This brain model uses 32-bit indices; a browser with WebGL 1 uint-index support is required');
          const localMin=[Infinity,Infinity,Infinity],localMax=[-Infinity,-Infinity,-Infinity];
          for(let i=0;i<positions.length;i+=3){const local=[positions[i],positions[i+1],positions[i+2]],point=transformPoint(nodeMatrix,local);for(let a=0;a<3;a++){min[a]=Math.min(min[a],point[a]);max[a]=Math.max(max[a],point[a]);localMin[a]=Math.min(localMin[a],local[a]);localMax[a]=Math.max(localMax[a],local[a]);}}
          const localCenter=localMin.map((v,i)=>(v+localMax[i])/2);
          const material=json.materials?.[primitive.material]||{};
          const pbr=material.pbrMetallicRoughness||{};
          const [baseMap,mrMap,normalMap]=await Promise.all([
            loadTextureIndex(json,binary,pbr.baseColorTexture?.index,textureCache).catch(()=>null),
            loadTextureIndex(json,binary,pbr.metallicRoughnessTexture?.index,textureCache).catch(()=>null),
            loadTextureIndex(json,binary,material.normalTexture?.index,textureCache).catch(()=>null)
          ]);
          const primitiveName=node.name||mesh.name||'Brain anatomy';
          parts.push({
            name:primitiveName,group:classifyPart(primitiveName),pBuffer,nBuffer,uvBuffer,tBuffer,iBuffer,count:indices.length,indexType,
            center:localCenter,nodeMatrix,baseMap:overrides.base||baseMap,metalRoughMap:mrMap,
            metalMap:overrides.metal,roughMap:overrides.rough,normalMap:overrides.normal||normalMap,
            baseColor:pbr.baseColorFactor||[1,1,1,1],metallic:pbr.metallicFactor??0,roughness:pbr.roughnessFactor??.82,
            alphaMode:material.alphaMode||'OPAQUE',alphaCutoff:material.alphaCutoff??.5
          });
        }
      }
      for (const child of (node.children||[])) await visit(child,nodeMatrix);
    }
    for (const index of roots) await visit(index,identity());
    if(!parts.length)throw new Error('The GLB contains no triangle mesh primitives');
    modelCenter=[(min[0]+max[0])/2,(min[1]+max[1])/2,(min[2]+max[2])/2];
    const maxExtent=Math.max(max[0]-min[0],max[1]-min[1],max[2]-min[2]);
    if(!Number.isFinite(maxExtent)||maxExtent<=0)throw new Error('The brain model has invalid dimensions');
    fitScale=2.25/maxExtent;
    parts.forEach(part=>{
      const localCenter=part.center;
      const c=transformPoint(part.nodeMatrix,localCenter);
      part.fitCenter=c.map((v,i)=>(v-modelCenter[i])*fitScale);
    });
    loaded=true;
    setStatus('BRAIN ANATOMY MODEL ONLINE','ready');
    window.setTimeout(()=>setStatus('', ''),2300);
  }

  function resize(){
    const rect=stage.getBoundingClientRect(),ratio=Math.min(window.devicePixelRatio||1,2);
    const w=Math.max(2,Math.floor(rect.width*ratio)),h=Math.max(2,Math.floor(rect.height*ratio));
    if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
    gl.viewport(0,0,w,h);
  }
  function bindAttribute(buffer,location,size){
    if(location<0)return;gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,size,gl.FLOAT,false,0,0);
  }
  function bindTexture(texture,unit,uniform,flagUniform,enabled){
    gl.activeTexture(gl.TEXTURE0+unit);gl.bindTexture(gl.TEXTURE_2D,texture||defaults.white);
    gl.uniform1i(loc[uniform],unit);gl.uniform1f(loc[flagUniform],enabled&&texture?1:0);
  }
  function render(){
    if(!running)return;
    resize();
    const width=canvas.width,height=canvas.height;
    gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    if(!loaded){window.requestAnimationFrame(render);return;}
    gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
    gl.enable(gl.CULL_FACE);gl.cullFace(gl.BACK);
    if(!dragging){targetY+=.0007;targetX+=(pointerY*.05-targetX)*.0007;}
    rotationX+=(targetX-rotationX)*.035;rotationY+=(targetY-rotationY)*.035;
    const projection=perspective(Math.PI/4,width/Math.max(1,height),.1,100);
    const view=translate(0,0,-3.6/zoom);
    const global=multiply(rotateX(rotationX),rotateY(rotationY));
    const fit=multiply(scale(fitScale,fitScale,fitScale),translate(-modelCenter[0],-modelCenter[1],-modelCenter[2]));
    parts.forEach(part=>{
      if(!layerState[part.group])return;
      const offset=exploded?translate(part.fitCenter[0]*.10,part.fitCenter[1]*.10,part.fitCenter[2]*.10):identity();
      const model=multiply(global,multiply(offset,multiply(fit,part.nodeMatrix)));
      const mvp=multiply(projection,multiply(view,model));
      bindAttribute(part.pBuffer,loc.aPosition,3);bindAttribute(part.nBuffer,loc.aNormal,3);bindAttribute(part.uvBuffer,loc.aUV,2);bindAttribute(part.tBuffer,loc.aTangent,4);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,part.iBuffer);
      gl.uniformMatrix4fv(loc.uMVP,false,mvp);gl.uniformMatrix4fv(loc.uModel,false,model);
      gl.uniform4fv(loc.uBaseColor,new Float32Array(part.baseColor));
      bindTexture(part.baseMap,0,'uBaseMap','uHasBaseMap',true);
      bindTexture(part.metalRoughMap,1,'uMetalRoughMap','uHasMetalRoughMap',true);
      bindTexture(part.metalMap,2,'uMetalMap','uHasMetalMap',!!part.metalMap);
      bindTexture(part.roughMap,3,'uRoughMap','uHasRoughMap',!!part.roughMap);
      bindTexture(part.normalMap,4,'uNormalMap','uHasNormalMap',true);
      gl.uniform1f(loc.uMetallic,part.metallic);gl.uniform1f(loc.uRoughness,part.roughness);
      gl.uniform1f(loc.uOpacity,transparent?.68:1);gl.uniform1f(loc.uCutaway,cutaway?1:0);gl.uniform1f(loc.uMaterialView,materialView);
      gl.uniform1f(loc.uAlphaCutoff,part.alphaMode==='MASK'?part.alphaCutoff:0.001);
      gl.drawElements(gl.TRIANGLES,part.count,part.indexType,0);
    });
    window.requestAnimationFrame(render);
  }
  canvas.addEventListener('contextmenu',event=>event.preventDefault());
  canvas.addEventListener('pointerdown',event=>{dragging=true;lastX=event.clientX;lastY=event.clientY;canvas.setPointerCapture(event.pointerId);});
  canvas.addEventListener('pointermove',event=>{
    const rect=canvas.getBoundingClientRect();pointerY=((event.clientY-rect.top)/rect.height-.5)*2;
    if(!dragging)return;targetY+=(event.clientX-lastX)*.008;targetX+=(event.clientY-lastY)*.006;lastX=event.clientX;lastY=event.clientY;
  });
  ['pointerup','pointercancel','lostpointercapture'].forEach(name=>canvas.addEventListener(name,()=>{dragging=false;}));
  canvas.addEventListener('wheel',event=>{event.preventDefault();zoom=Math.max(.58,Math.min(2.2,zoom*(event.deltaY>0?.93:1.07)));},{passive:false});
  canvas.addEventListener('dblclick',()=>{targetX=-.12;targetY=selectedRegion==='Occipital lobe'?2.75:selectedRegion==='Temporal lobe'?-.2:-.45;zoom=1.1;});
  window.addEventListener('resize',resize);
  render();
  loadModel().catch(error=>{
    console.error('T-Scanner GLB brain model could not be loaded:',error);
    setStatus('MODEL NOT LOADED · CHECK backend/3d/brain FOLDER','error');
    toast(error.message||'Unable to load the 3D brain model.',6500);
  });
  return {
    reset(){targetX=-.12;targetY=-.45;zoom=1;},
    explode(){if(parts.length<2){toast('Explode view requires separate anatomical meshes in the GLB model.');return false;}exploded=!exploded;return exploded;},
    section(){cutaway=!cutaway;return cutaway;},
    transparency(){transparent=!transparent;return transparent;},
    selectRegion(region){selectedRegion=region;targetY=region==='Occipital lobe'?2.75:region==='Temporal lobe'?-.2:-.45;},
    setMaterialFilter(name){if(!(name in materialFilterValues))return;materialView=materialFilterValues[name];},
    toggleLayer(name,visible){if(!parts.some(part=>part.group===name)){toast('This GLB does not contain a separate '+name+' mesh. The model may be a single merged object.');return layerState[name];}layerState[name]=Boolean(visible);return layerState[name];},
    toggleTumorHighlight(visible){tumorHighlight=Boolean(visible);if(visible)toast('The anatomy model is not registered to the uploaded MRI; no patient tumour position is displayed.');return tumorHighlight;}
  };
})();

if (!BrainRenderer) {
  const canvas = $('#brainCanvas');
  if (canvas) canvas.style.display = 'none';
}

$('#brainReset').addEventListener('click', () => BrainRenderer?.reset());
$('#brainExplode').addEventListener('click', event => { const on = BrainRenderer?.explode(); event.currentTarget.classList.toggle('active', !!on); });
$('#brainSection').addEventListener('click', event => { const on = BrainRenderer?.section(); event.currentTarget.classList.toggle('active', !!on); });
$('#brainFilters').addEventListener('click', () => $('#brainFilterPanel').classList.toggle('hidden'));
$$('[data-brain-filter]').forEach(button => button.addEventListener('click', () => {
  const filter = button.dataset.brainFilter;
  BrainRenderer?.setMaterialFilter(filter);
  $$('.brain-filter-option').forEach(option => option.classList.toggle('active', option === button));
}));
$('#layerTransparency').addEventListener('click', event => { const on = BrainRenderer?.transparency(); event.currentTarget.classList.toggle('active', !!on); });
const anatomyCopy = {
  'Frontal lobe': 'The frontal lobes contribute to planning, decision-making, voluntary movement, attention and aspects of language and behaviour.',
  'Parietal lobe': 'The parietal lobes integrate sensory information and contribute to spatial awareness, body position and attention.',
  'Temporal lobe': 'The temporal lobes are involved in hearing, memory, emotion and language comprehension.',
  'Occipital lobe': 'The occipital lobes are the main cortical region for processing visual information.',
  'Cerebellum': 'The cerebellum contributes to movement coordination, balance, motor learning and timing.',
  'Brainstem': 'The brainstem connects the brain with the spinal cord and supports essential functions such as breathing, alertness and heart-rate regulation.'
};
function selectRegion(region) {
  state.region = region;
  BrainRenderer?.selectRegion(region);
  $('#anatomyTitle').textContent = region;
  $('#anatomyDescription').textContent = anatomyCopy[region] || '';
  $('#anatomyCard').classList.remove('hidden');
  $$('[data-region-select]').forEach(button => button.classList.toggle('active', button.dataset.regionSelect === region));
}
$$('[data-region-select]').forEach(button => button.addEventListener('click', () => selectRegion(button.dataset.regionSelect)));
$$('[data-region]').forEach(button => button.addEventListener('click', () => selectRegion(button.dataset.region)));
$('#closeAnatomy').addEventListener('click', () => $('#anatomyCard').classList.add('hidden'));

// ---------------------------------------------------------------------------
// Scanner chamber and analysis requests.
// ---------------------------------------------------------------------------
const scanInput = $('#scanFiles');
const dropzone = $('#dropzone');
function setFiles(files) {
  state.files = Array.from(files || []).filter(file => file && file.size > 0);
  $('#selectedFiles').innerHTML = '';
  if (!state.files.length) {
    $('#selectedFiles').innerHTML = '<span class="empty-copy">No files selected</span>';
  } else {
    state.files.forEach(file => {
      const row = document.createElement('div'); row.className = 'selected-file';
      const name = document.createElement('span'); name.textContent = file.name;
      const size = document.createElement('small'); size.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
      row.append(name, size); $('#selectedFiles').append(row);
    });
  }
  $('#fileState').textContent = state.files.length ? `${state.files.length} IMAGE${state.files.length === 1 ? '' : 'S'} READY` : 'AWAITING IMAGE';
  $('#startScan').disabled = !state.files.length;
}
scanInput.addEventListener('change', event => setFiles(event.target.files));
['dragenter', 'dragover'].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.add('dragover'); }));
['dragleave', 'drop'].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.remove('dragover'); }));
dropzone.addEventListener('drop', event => setFiles(event.dataTransfer.files));

let progressTimer = null;
let activeScanPreviewUrl = null;
let scanStartedAt = 0;
let scanSweepFrame = 0;
let scanSweepActive = false;
const SCAN_MINIMUM_DURATION_MS = 5600;

// Drive the search reticle from JavaScript instead of relying on CSS keyframes.
// This makes the box visibly traverse every horizontal band of the uploaded MRI.
function stopScanSweep() {
  scanSweepActive = false;
  if (scanSweepFrame) window.cancelAnimationFrame(scanSweepFrame);
  scanSweepFrame = 0;
  const reticle = $('#scanSearchReticle');
  reticle?.classList.remove('is-sweeping');
}
function startScanSweep() {
  stopScanSweep();
  const frame = $('.scan-preview-frame');
  const reticle = $('#scanSearchReticle');
  const label = $('#scanReticleLabel');
  if (!frame || !reticle) return;

  scanSweepActive = true;
  reticle.classList.add('is-sweeping');
  reticle.style.left = '2%';
  reticle.style.top = '2%';
  const cycleMs = 5200;
  const rowDuration = cycleMs / 4;
  const maxLeft = 66;
  const rowTops = [2, 26, 50, 74];

  const tick = timestamp => {
    if (!scanSweepActive || !reticle.isConnected) return;
    const elapsed = (timestamp - scanStartedAt) % cycleMs;
    const row = Math.min(3, Math.floor(elapsed / rowDuration));
    const rowProgress = (elapsed - row * rowDuration) / rowDuration;
    // Smooth left-to-right / right-to-left raster sweep across four bands.
    const eased = rowProgress * rowProgress * (3 - 2 * rowProgress);
    const x = row % 2 === 0 ? 2 + eased * maxLeft : 2 + (1 - eased) * maxLeft;
    reticle.style.left = `${x.toFixed(2)}%`;
    reticle.style.top = `${rowTops[row]}%`;
    reticle.style.width = '32%';
    reticle.style.height = '22%';
    if (label) label.textContent = `SEARCHING · BAND ${row + 1}/4`;
    $('#scanPreviewState').textContent = `FULL-FIELD SEARCH · ${row + 1}/4`;
    scanSweepFrame = window.requestAnimationFrame(tick);
  };
  scanSweepFrame = window.requestAnimationFrame(tick);
}

function progressStart(previewFile = null) {
  scanStartedAt = performance.now();
  const scannerVisual = $('.scanner-visual');
  scannerVisual.classList.add('scanning');
  const previewShell = $('#scanPreviewShell');
  if (previewFile && previewShell) {
    if (activeScanPreviewUrl) URL.revokeObjectURL(activeScanPreviewUrl);
    activeScanPreviewUrl = URL.createObjectURL(previewFile);
    const previewImage = $('#scanPreviewImage');
    previewImage.src = activeScanPreviewUrl;
    previewShell.classList.add('active');
    $('#scanPreviewState').textContent = 'INITIALIZING FULL-FIELD SEARCH';
    // Start on the next paint so the image and active preview are already visible.
    window.requestAnimationFrame(() => startScanSweep());
  }
  $('#scanProgress').classList.remove('hidden');
  $('#scanError').classList.add('hidden');
  $('#startScan').disabled = true;
  $('#scannerBeam').parentElement.classList.add('scanning');
  $('#progressPercent').textContent = '3%';
  $('#progressBar').style.width = '3%';
  let progress = 3;
  const labels = [[0, 'PREPARING IMAGE'], [24, 'RUNNING CLASSIFICATION'], [57, 'GENERATING EXPLANATION'], [80, 'ARCHIVING CASE'], [95, 'FINALIZING RESULT']];
  window.clearInterval(progressTimer);
  progressTimer = window.setInterval(() => {
    progress = Math.min(94, progress + 1.3 + Math.random() * 2.2);
    $('#progressPercent').textContent = `${Math.round(progress)}%`;
    $('#progressBar').style.width = `${progress}%`;
    const label = labels.filter(([threshold]) => progress >= threshold).at(-1)?.[1] || labels[0][1];
    $('#progressLabel').textContent = label;
  }, 350);
}
async function finishScanAnimation() {
  // Keep the scan visible long enough for at least one complete multi-band sweep.
  const elapsed = performance.now() - scanStartedAt;
  const remaining = SCAN_MINIMUM_DURATION_MS - elapsed;
  if (remaining > 0) await new Promise(resolve => window.setTimeout(resolve, remaining));
  progressEnd();
}
function progressEnd() {
  window.clearInterval(progressTimer);
  stopScanSweep();
  $('#progressPercent').textContent = '100%'; $('#progressBar').style.width = '100%';
  $('#progressLabel').textContent = 'COMPLETE';
  $('#scannerBeam').parentElement.classList.remove('scanning');
  $('#scanPreviewState').textContent = 'ANALYSIS COMPLETE';
  const label = $('#scanReticleLabel');
  if (label) label.textContent = 'SEARCH COMPLETE';
  window.setTimeout(() => {
    $('#scanPreviewShell')?.classList.remove('active');
    if (activeScanPreviewUrl) { URL.revokeObjectURL(activeScanPreviewUrl); activeScanPreviewUrl = null; }
    $('#scanPreviewImage')?.removeAttribute('src');
  }, 900);
  $('#startScan').disabled = !state.files.length;
  window.setTimeout(() => $('#scanProgress').classList.add('hidden'), 1500);
}

$('#startScan').addEventListener('click', async () => {
  if (!state.files.length) return;
  progressStart(state.files[0] || null);
  try {
    const mode = $('#analysisMode').value;
    if (mode === 'batch' && state.files.length > 1) {
      const form = new FormData();
      state.files.forEach(file => form.append('files', file));
      const data = await api('/api/analyze/batch', { method: 'POST', body: form });
      const ok = (data.results || []).filter(result => result.status === 'success');
      await finishScanAnimation();
      toast(`${ok.length} of ${data.count} scan(s) analyzed and archived.`);
      if (ok.length) {
        state.result = ok[0]; state.explanation = null; state.caseId = ok[0].case_id; state.currentFile = state.files[0];
        state.annotatedUrl = ok[0].annotated_image ? `/api/annotated/${encodeURIComponent(ok[0].annotated_image)}` : '';
        state.annotatedImage = null; state.heatmapImage = null; state.heatmapUrl = '';
        loadCurrentImage(state.currentFile);
        renderResult(ok[0], null);
        setRoom('explore');
        void runSimilarSearch({ silent: true });
      }
      return;
    }
    const form = new FormData(); form.append('file', state.files[0]);
    const data = await api('/api/explain', { method: 'POST', body: form });
    state.result = data.classification;
    state.explanation = data.explanation;
    state.caseId = data.case_id || data.classification?.case_id || null;
    state.currentFile = state.files[0];
    const active = $('#sequenceSelect').value;
    if (active === 'coronal' || active === 'sagittal') toast('Multi-planar reconstruction requires a volumetric source; current scan is 2D.');
    loadCurrentImage(state.currentFile);
    state.annotatedUrl = state.result.annotated_image ? `/api/annotated/${encodeURIComponent(state.result.annotated_image)}` : '';
    state.annotatedImage = null; state.heatmapImage = null; state.heatmapUrl = '';
    renderResult(state.result, state.explanation);
    await finishScanAnimation();
    setRoom('explore');
    toast('AI analysis completed. The scan has been saved in the local case archive.');
    void runSimilarSearch({ silent: true });
  } catch (error) {
    await finishScanAnimation();
    $('#scanError').textContent = error.message;
    $('#scanError').classList.remove('hidden');
    toast(`Analysis failed: ${error.message}`, 4500);
  }
});

// ---------------------------------------------------------------------------
// MRI inspection canvas: image filtering, pan/zoom, pixel probe, ROI and ruler.
// ---------------------------------------------------------------------------
const mriCanvas = $('#mriCanvas');
const mriContext = mriCanvas.getContext('2d', { willReadFrequently: true });
const offscreen = document.createElement('canvas');
const offscreenContext = offscreen.getContext('2d', { willReadFrequently: true });
let filteredCanvas = document.createElement('canvas');
let filteredContext = filteredCanvas.getContext('2d');
let drawInfo = null;
function loadCurrentImage(file) {
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(file);
  const image = new Image();
  image.onload = () => {
    state.image = image; state.zoom = 1; state.panX = 0; state.panY = 0; state.measurePoints = []; state.roi = null;
    $('#viewerPlaceholder').classList.add('hidden');
    $('#viewerFilename').textContent = file.name.toUpperCase();
    $('#viewerDimensions').textContent = `${image.naturalWidth} × ${image.naturalHeight}`;
    $('#imageSize').textContent = `${image.naturalWidth}×${image.naturalHeight}`;
    $('#findSimilar').disabled = false; $('#makeGradcam').disabled = !state.caseId; $('#printReport').disabled = !state.result;
    drawMRI();
  };
  image.src = state.objectUrl;
}
function currentViewerImage() {
  if (state.viewerMode === 'annotated' && state.annotatedImage) return state.annotatedImage;
  if (state.viewerMode === 'heatmap' && state.heatmapImage) return state.heatmapImage;
  return state.image;
}
function createFilteredCanvas(image) {
  const maxDimension = 2600;
  const ratio = Math.min(1, maxDimension / Math.max(image.naturalWidth, image.naturalHeight));
  const width = Math.max(1, Math.round(image.naturalWidth * ratio));
  const height = Math.max(1, Math.round(image.naturalHeight * ratio));
  offscreen.width = width; offscreen.height = height;
  offscreenContext.clearRect(0, 0, width, height);
  offscreenContext.drawImage(image, 0, 0, width, height);
  const pixels = offscreenContext.getImageData(0, 0, width, height);
  const data = pixels.data;
  const contrast = state.contrast / 100, brightness = state.brightness / 255, gamma = Math.max(.2, state.gamma);
  for (let index = 0; index < data.length; index += 4) {
    let r = data[index] / 255, g = data[index + 1] / 255, b = data[index + 2] / 255;
    if (state.grayscale) { const gray = .2126 * r + .7152 * g + .0722 * b; r = g = b = gray; }
    const adjust = value => {
      value = (value - .5) * contrast + .5 + brightness;
      value = Math.pow(Math.max(0, Math.min(1, value)), 1 / gamma);
      if (state.invert) value = 1 - value;
      return Math.max(0, Math.min(1, value)) * 255;
    };
    data[index] = adjust(r); data[index + 1] = adjust(g); data[index + 2] = adjust(b);
  }
  offscreenContext.putImageData(pixels, 0, 0);
  filteredCanvas.width = width; filteredCanvas.height = height;
  filteredContext.clearRect(0, 0, width, height); filteredContext.drawImage(offscreen, 0, 0);
  return filteredCanvas;
}
function drawMRI() {
  if (!mriCanvas || !mriContext) return;
  const rect = $('#mriViewer').getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(2, Math.round(rect.width * dpr)); const height = Math.max(2, Math.round(rect.height * dpr));
  if (mriCanvas.width !== width || mriCanvas.height !== height) { mriCanvas.width = width; mriCanvas.height = height; }
  mriContext.setTransform(dpr, 0, 0, dpr, 0, 0);
  mriContext.clearRect(0, 0, rect.width, rect.height);
  const image = currentViewerImage();
  if (!image) { drawInfo = null; return; }
  const filtered = createFilteredCanvas(image);
  const scale = Math.min(rect.width / filtered.width, rect.height / filtered.height) * state.zoom;
  const drawWidth = filtered.width * scale, drawHeight = filtered.height * scale;
  const x = (rect.width - drawWidth) / 2 + state.panX;
  const y = (rect.height - drawHeight) / 2 + state.panY;
  drawInfo = { x, y, scale, width: filtered.width, height: filtered.height, naturalWidth: image.naturalWidth, naturalHeight: image.naturalHeight, rectWidth: rect.width, rectHeight: rect.height };
  mriContext.save();
  mriContext.imageSmoothingEnabled = true; mriContext.imageSmoothingQuality = 'high';
  if (state.compareBeforeAfter && state.viewerMode === 'original') {
    mriContext.drawImage(image, x, y, drawWidth, drawHeight);
    const split = Math.max(0, Math.min(100, state.compareSplit)) / 100;
    mriContext.save(); mriContext.beginPath(); mriContext.rect(x, y, drawWidth * split, drawHeight); mriContext.clip();
    mriContext.drawImage(filtered, x, y, drawWidth, drawHeight); mriContext.restore();
    mriContext.strokeStyle = '#7cf1ed'; mriContext.lineWidth = 1; mriContext.beginPath(); mriContext.moveTo(x + drawWidth * split, y); mriContext.lineTo(x + drawWidth * split, y + drawHeight); mriContext.stroke();
  } else {
    mriContext.drawImage(filtered, x, y, drawWidth, drawHeight);
  }
  if (state.viewerMode === 'original' && state.result?.detections?.length) {
    state.result.detections.forEach(detection => {
      const box = detection.bbox; if (!box) return;
      const bx = x + box.x1 * scale * (filtered.width / image.naturalWidth);
      const by = y + box.y1 * scale * (filtered.height / image.naturalHeight);
      const bw = (box.x2 - box.x1) * scale * (filtered.width / image.naturalWidth);
      const bh = (box.y2 - box.y1) * scale * (filtered.height / image.naturalHeight);
      mriContext.strokeStyle = '#79f2d0'; mriContext.lineWidth = 2; mriContext.strokeRect(bx, by, bw, bh);
      mriContext.fillStyle = 'rgba(5,18,25,.82)'; mriContext.fillRect(bx, Math.max(0, by - 18), Math.max(62, bw), 17);
      mriContext.fillStyle = '#a7ffe8'; mriContext.font = '9px monospace'; mriContext.fillText(`${detection.class || 'Region'} ${(detection.confidence * 100).toFixed(1)}%`, bx + 5, Math.max(12, by - 6));
    });
  }
  if (state.measurePoints.length) {
    mriContext.strokeStyle = '#7cf1ed'; mriContext.fillStyle = '#7cf1ed'; mriContext.lineWidth = 1.5;
    state.measurePoints.forEach(point => { mriContext.beginPath(); mriContext.arc(x + point.x * scale, y + point.y * scale, 3, 0, Math.PI * 2); mriContext.fill(); });
    if (state.measurePoints.length === 2) {
      const [a, b] = state.measurePoints; const sx = scale * filtered.width / image.naturalWidth; const sy = scale * filtered.height / image.naturalHeight; mriContext.beginPath(); mriContext.moveTo(x + a.x * sx, y + a.y * sy); mriContext.lineTo(x + b.x * sx, y + b.y * sy); mriContext.stroke();
      const distance = Math.hypot(b.x - a.x, b.y - a.y); mriContext.fillText(`${distance.toFixed(1)} px`, x + (a.x + b.x) * sx / 2 + 6, y + (a.y + b.y) * sy / 2 - 6);
    }
  }
  if (state.roi) {
    const roi = state.roi; const sx = scale * filtered.width / image.naturalWidth; const sy = scale * filtered.height / image.naturalHeight; mriContext.strokeStyle = '#ffd28b'; mriContext.lineWidth = 1.5; mriContext.setLineDash([5, 4]);
    mriContext.strokeRect(x + roi.x * sx, y + roi.y * sy, roi.w * sx, roi.h * sy); mriContext.setLineDash([]);
  }
  mriContext.restore();
  $('#zoomReadout').textContent = `${Math.round(state.zoom * 100)}%`;
}
function canvasPoint(event) { const rect = mriCanvas.getBoundingClientRect(); return { x: event.clientX - rect.left, y: event.clientY - rect.top }; }
function toImagePoint(point) {
  if (!drawInfo) return null;
  const x = (point.x - drawInfo.x) / drawInfo.scale, y = (point.y - drawInfo.y) / drawInfo.scale;
  if (x < 0 || y < 0 || x >= drawInfo.width || y >= drawInfo.height) return null;
  return { x: x * drawInfo.naturalWidth / drawInfo.width, y: y * drawInfo.naturalHeight / drawInfo.height };
}
mriCanvas.addEventListener('pointerdown', event => {
  if (!state.image) return;
  const point = canvasPoint(event); const imagePoint = toImagePoint(point);
  if ($('#measureToggle').classList.contains('active')) {
    if (!imagePoint) return;
    state.measurePoints.push(imagePoint);
    if (state.measurePoints.length > 2) state.measurePoints = [imagePoint];
    if (state.measurePoints.length === 2) toast(`Measurement: ${Math.hypot(state.measurePoints[1].x-state.measurePoints[0].x, state.measurePoints[1].y-state.measurePoints[0].y).toFixed(1)} pixels. No spatial calibration is available.`);
    drawMRI(); return;
  }
  if ($('#roiToggle').classList.contains('active')) {
    if (!imagePoint) return;
    state.pointerStart = imagePoint; state.roi = { x: imagePoint.x, y: imagePoint.y, w: 0, h: 0 };
    mriCanvas.setPointerCapture(event.pointerId); drawMRI(); return;
  }
  state.isPanning = true; state.pointerStart = point; mriCanvas.setPointerCapture(event.pointerId);
});
mriCanvas.addEventListener('pointermove', event => {
  const point = canvasPoint(event); const imagePoint = toImagePoint(point);
  if (imagePoint && drawInfo && state.image) {
    $('#cursorReadout').textContent = `${Math.round(imagePoint.x)} / ${Math.round(imagePoint.y)}`;
    const probeContext = filteredCanvas.getContext('2d', { willReadFrequently: true });
    const pixel = probeContext.getImageData(Math.max(0, Math.min(filteredCanvas.width - 1, Math.floor(imagePoint.x * filteredCanvas.width / state.image.naturalWidth))), Math.max(0, Math.min(filteredCanvas.height - 1, Math.floor(imagePoint.y * filteredCanvas.height / state.image.naturalHeight))), 1, 1).data;
    $('#pixelReadout').textContent = `PIXEL: ${Math.round(imagePoint.x)}, ${Math.round(imagePoint.y)} · INTENSITY: ${Math.round((pixel[0] + pixel[1] + pixel[2]) / 3)}`;
  } else { $('#cursorReadout').textContent = '—'; $('#pixelReadout').textContent = 'PIXEL: — · INTENSITY: —'; }
  if (state.roi && state.pointerStart && $('#roiToggle').classList.contains('active')) {
    const next = toImagePoint(point); if (!next) return;
    const start = state.pointerStart;
    state.roi = { x: Math.min(start.x, next.x), y: Math.min(start.y, next.y), w: Math.abs(next.x - start.x), h: Math.abs(next.y - start.y) };
    drawMRI(); return;
  }
  if (state.isPanning && state.pointerStart) {
    state.panX += point.x - state.pointerStart.x; state.panY += point.y - state.pointerStart.y; state.pointerStart = point; drawMRI();
  }
});
mriCanvas.addEventListener('pointerup', () => {
  if (state.roi && state.roi.w > 4 && state.roi.h > 4) {
    const area = state.roi.w * state.roi.h; $('#exportRoi').disabled = false; toast(`Region selected: ${Math.round(state.roi.w)} × ${Math.round(state.roi.h)} px (${Math.round(area).toLocaleString()} px²).`);
  }
  state.pointerStart = null; state.isPanning = false;
});
mriCanvas.addEventListener('pointercancel', () => { state.pointerStart = null; state.isPanning = false; });
$('#mriViewer').addEventListener('wheel', event => { if (!state.image) return; event.preventDefault(); state.zoom = Math.max(.5, Math.min(5, state.zoom * (event.deltaY < 0 ? 1.12 : .89))); drawMRI(); }, { passive: false });
function bindRange(id, key, display, divisor = 1) {
  $(`#${id}`).addEventListener('input', event => { state[key] = Number(event.target.value) / divisor; $(`#${display}`).textContent = key === 'gamma' ? state[key].toFixed(2) : String(event.target.value); drawMRI(); });
}
bindRange('brightness', 'brightness', 'brightnessValue'); bindRange('contrast', 'contrast', 'contrastValue'); bindRange('gamma', 'gamma', 'gammaValue', 100);
$('#invertToggle').addEventListener('click', event => { state.invert = !state.invert; event.currentTarget.classList.toggle('active', state.invert); drawMRI(); });
$('#grayToggle').addEventListener('click', event => { state.grayscale = !state.grayscale; event.currentTarget.classList.toggle('active', state.grayscale); drawMRI(); });
$('#measureToggle').addEventListener('click', event => { event.currentTarget.classList.toggle('active'); $('#roiToggle').classList.remove('active'); state.roi = null; state.measurePoints = []; });
$('#roiToggle').addEventListener('click', event => { event.currentTarget.classList.toggle('active'); $('#measureToggle').classList.remove('active'); state.measurePoints = []; state.roi = null; });
$('#exportRoi').addEventListener('click', () => {
  if (!state.image || !state.roi || state.roi.w < 2 || state.roi.h < 2) return;
  const source = currentViewerImage();
  const sx = Math.round(state.roi.x * source.naturalWidth / drawInfo.naturalWidth);
  const sy = Math.round(state.roi.y * source.naturalHeight / drawInfo.naturalHeight);
  const sw = Math.round(state.roi.w * source.naturalWidth / drawInfo.naturalWidth);
  const sh = Math.round(state.roi.h * source.naturalHeight / drawInfo.naturalHeight);
  const crop = document.createElement('canvas'); crop.width = Math.max(1, sw); crop.height = Math.max(1, sh);
  crop.getContext('2d').drawImage(source, sx, sy, sw, sh, 0, 0, crop.width, crop.height);
  const link = document.createElement('a'); link.href = crop.toDataURL('image/png'); link.download = 'tscanner-selected-roi.png'; link.click();
});
$('#compareToggle').addEventListener('click', event => {
  state.compareBeforeAfter = !state.compareBeforeAfter;
  event.currentTarget.classList.toggle('active', state.compareBeforeAfter);
  $('#compareSlider').classList.toggle('hidden', !state.compareBeforeAfter);
  drawMRI();
});
$('#compareRange').addEventListener('input', event => { state.compareSplit = Number(event.target.value); $('#compareRangeValue').textContent = `${state.compareSplit}%`; drawMRI(); });
$('#resetView').addEventListener('click', () => {
  state.brightness = 0; state.contrast = 100; state.gamma = 1; state.invert = false; state.grayscale = true; state.zoom = 1; state.panX = 0; state.panY = 0; state.roi = null; state.measurePoints = []; state.compareBeforeAfter = false; state.compareSplit = 50;
  $('#exportRoi').disabled = true; $('#compareToggle').classList.remove('active'); $('#compareSlider').classList.add('hidden'); $('#compareRange').value = 50; $('#compareRangeValue').textContent = '50%';
  $('#brightness').value = 0; $('#brightnessValue').textContent = '0'; $('#contrast').value = 100; $('#contrastValue').textContent = '100'; $('#gamma').value = 100; $('#gammaValue').textContent = '1.00';
  $('#invertToggle').classList.remove('active'); $('#grayToggle').classList.add('active'); $('#measureToggle').classList.remove('active'); $('#roiToggle').classList.remove('active'); drawMRI();
});
function setViewerMode(mode) {
  state.viewerMode = mode;
  $$('.viewer-actions button').forEach(button => button.classList.toggle('active', button.id === ({ original: 'viewOriginal', annotated: 'viewAnnotated', heatmap: 'viewHeatmap' }[mode])));
  drawMRI();
}
$('#viewOriginal').addEventListener('click', () => setViewerMode('original'));
$('#viewAnnotated').addEventListener('click', () => {
  if (!state.annotatedUrl) return toast('Annotated output is not available for this scan.');
  if (!state.annotatedImage) {
    state.annotatedImage = new Image(); state.annotatedImage.onload = () => { setViewerMode('annotated'); }; state.annotatedImage.onerror = () => toast('Could not load annotated MRI output.'); state.annotatedImage.src = state.annotatedUrl;
  } else setViewerMode('annotated');
});
$('#viewHeatmap').addEventListener('click', () => { if (state.heatmapImage) setViewerMode('heatmap'); });
$('#downloadView').addEventListener('click', () => {
  const link = document.createElement('a'); link.href = mriCanvas.toDataURL('image/png'); link.download = `tscanner-${state.viewerMode}-view.png`; link.click();
});
$('#thresholdRange').addEventListener('input', event => { $('#thresholdValue').textContent = `${event.target.value}%`; });
$('#thresholdRange').addEventListener('change', async event => {
  try {
    const data = await api('/api/settings/confidence', { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ value: Number(event.target.value) / 100 }) });
    toast(`Confidence threshold set to ${(data.confidence_threshold * 100).toFixed(0)}% for future scans.`);
  } catch (error) { toast(error.message); }
});
$('#makeGradcam').addEventListener('click', async () => {
  if (!state.caseId) return;
  $('#makeGradcam').disabled = true;
  try {
    const data = await api(`/api/gradcam/${encodeURIComponent(state.caseId)}`, { method: 'POST' });
    if (!data.available) { toast(data.reason || 'Grad-CAM is unavailable for this model.'); return; }
    state.heatmapUrl = data.image_url;
    state.heatmapImage = new Image();
    state.heatmapImage.onload = () => { $('#viewHeatmap').disabled = false; setViewerMode('heatmap'); toast('Grad-CAM generated. This represents model attention, not tumour segmentation.'); };
    state.heatmapImage.onerror = () => toast('Grad-CAM output could not be loaded.');
    state.heatmapImage.src = `${data.image_url}?t=${Date.now()}`;
  } catch (error) { toast(error.message); }
  finally { $('#makeGradcam').disabled = false; }
});

function renderResult(result, explanation) {
  $('#viewerFilename').textContent = (result.filename || state.currentFile?.name || 'MRI SCAN').toUpperCase();
  $('#resultBadge').textContent = result.tumor_detected ? 'TUMOUR FLAGGED' : 'NO TUMOUR CLASS';
  $('#resultBadge').classList.toggle('detected', !!result.tumor_detected);
  $('#resultBadge').classList.toggle('clear', !result.tumor_detected);
  $('#predictedType').textContent = result.predicted_type || 'Unknown';
  const confidence = result.best_confidence == null ? null : Number(result.best_confidence);
  $('#confidenceValue').textContent = confidence == null ? '—' : `${(confidence * 100).toFixed(1)}%`;
  $('#confidenceBar').style.width = `${Math.max(0, Math.min(100, (confidence || 0) * 100))}%`;
  $('#inferenceTime').textContent = result.processing_time_ms == null ? '—' : `${Math.round(result.processing_time_ms)} ms`;
  $('#detectionCount').textContent = String((result.detections || []).length);
  $('#imageSize').textContent = result.image_width ? `${result.image_width}×${result.image_height}` : '—';
  const predictions = result.top_predictions || result.class_probabilities || [];
  $('#probabilityList').innerHTML = predictions.length ? predictions.slice(0, 4).map(item => {
    const pct = Math.max(0, Math.min(100, Number(item.probability || 0) * 100));
    return `<div class="probability-row"><span>${escapeHTML(item.class || 'Class')}</span><div class="probability-track"><i style="width:${pct}%"></i></div><b>${pct.toFixed(1)}%</b></div>`;
  }).join('') : '<div class="empty-copy">The loaded model did not return a class probability vector.</div>';
  const detections = result.detections || [];
  const localizationLabel = result.localization_method === 'heuristic_estimate' ? 'HEURISTIC ESTIMATE' : result.localization_method === 'detector' ? 'DETECTOR OUTPUT' : 'NO LOCALIZATION';
  $('#detectionList').innerHTML = `<div class="empty-copy" style="margin-bottom:6px">${localizationLabel}</div>` + (detections.length ? detections.map((item, index) => `<div class="detection-row"><span>${index + 1}. ${escapeHTML(item.class || 'Region')}</span><small>${(Number(item.confidence || 0) * 100).toFixed(1)}%</small></div>`).join('') : '<div class="empty-copy">No localized bounding boxes were returned.</div>');
  $('#aiSummary').textContent = explanation?.summary || 'No generated explanation is available for this analysis. Review the model output with a qualified medical professional.';
  const precautions = explanation?.precautions || [];
  $('#aiPrecautions').innerHTML = precautions.length ? `<ul class="precaution-list">${precautions.slice(0, 4).map(item => `<li>${escapeHTML(item)}</li>`).join('')}</ul>` : '';
  $('#findSimilar').disabled = !state.currentFile;
  $('#makeGradcam').disabled = !state.caseId;
  $('#printReport').disabled = false;
  $('#compareCategory').textContent = String(result.predicted_type || 'UNKNOWN').toUpperCase();
  $('#runSimilar').disabled = !state.currentFile;
  $('#viewHeatmap').disabled = !state.heatmapImage;
  setViewerMode('original');
}
function escapeHTML(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
}

// ---------------------------------------------------------------------------
// Similar MRI case retrieval. Scores are calculated from real folder contents.
// ---------------------------------------------------------------------------
$('#findSimilar').addEventListener('click', () => { setRoom('compare'); runSimilarSearch(); });
$('#runSimilar').addEventListener('click', runSimilarSearch);
$('#openAllSimilar').addEventListener('click', () => { setRoom('compare'); if (state.similarity.length) renderSimilarCards(state.similarity); });
async function runSimilarSearch(options = {}) {
  const { silent = false } = options;
  if (!state.currentFile || !state.result || state.similarLoading) return;
  state.similarLoading = true;
  $('#runSimilar').disabled = true;
  const loadingMarkup = '<div class="comparison-empty"><div class="empty-orbit">⌕</div><strong>SEARCHING YOUR MRI DATASET</strong><p>Collecting real examples from the predicted class in the Train and Test folders...</p></div>';
  $('#similarGrid').innerHTML = loadingMarkup;
  $('#resultSimilarGrid').innerHTML = '<div class="empty-copy">Searching the matching tumour-class folders…</div>';
  const form = new FormData(); form.append('file', state.currentFile); form.append('predicted_type', state.result.predicted_type || 'unknown'); form.append('limit', '12');
  try {
    const data = await api('/api/cases/similar', { method: 'POST', body: form });
    state.similarity = data.results || [];
    if (!state.similarity.length) {
      const message = escapeHTML(data.message || 'No images were found in the selected dataset class.');
      const expected = `<br><br>Expected folders: <code>backend/dataset/Train/${escapeHTML(data.category || 'class')}/</code> and <code>backend/dataset/Test/${escapeHTML(data.category || 'class')}/</code>.`;
      $('#similarGrid').innerHTML = `<div class="comparison-empty"><div class="empty-orbit">⌕</div><strong>NO MRI EXAMPLES FOUND FOR THIS CLASS</strong><p>${message}${expected}</p></div>`;
      $('#resultSimilarGrid').innerHTML = `<div class="empty-copy">No same-class reference images were found. Check <code>backend/dataset/Train</code> and <code>backend/dataset/Test</code>.</div>`;
      $('#openAllSimilar').disabled = true;
    } else {
      renderSimilarCards(state.similarity);
      renderOutputSimilar(state.similarity);
      $('#openAllSimilar').disabled = false;
    }
    if (!silent) toast(`${state.similarity.length} same-class reference image(s) retrieved.`);
  } catch (error) {
    const message = escapeHTML(error.message);
    $('#similarGrid').innerHTML = `<div class="comparison-empty"><strong>REFERENCE SEARCH FAILED</strong><p>${message}</p></div>`;
    $('#resultSimilarGrid').innerHTML = `<div class="empty-copy">Reference images could not be loaded: ${message}</div>`;
    $('#openAllSimilar').disabled = true;
    if (!silent) toast(`Reference search failed: ${error.message}`, 4000);
  } finally {
    state.similarLoading = false;
    $('#runSimilar').disabled = false;
  }
}
function renderOutputSimilar(items) {
  const displayItems = items.slice(0, 3);
  $('#resultSimilarGrid').innerHTML = displayItems.map((item, index) => `<button class="result-similar-card" type="button" data-result-reference="${escapeHTML(item.id)}" aria-label="Open same-class MRI reference ${index + 1}"><img src="${escapeHTML(item.image_url)}" alt="Same-class MRI reference example" loading="lazy"><span><b>REFERENCE ${String(index + 1).padStart(2, '0')}</b><small>${escapeHTML(String(item.category || '').toUpperCase())} · ${(item.dataset_split || 'DATASET').toUpperCase()}</small></span></button>`).join('');
  $$('[data-result-reference]').forEach(button => button.addEventListener('click', () => {
    const item = state.similarity.find(match => match.id === button.dataset.resultReference);
    if (!item) return;
    setRoom('compare');
    renderSimilarCards(state.similarity);
    openComparison(item);
  }));
}
function renderSimilarCards(items, source = 'search') {
  $('#similarGrid').innerHTML = items.map((item, index) => {
    const isSaved = state.referenceFavorites.some(saved => saved.id === item.id && saved.category === item.category);
    const split = item.dataset_split || item.metadata?.split || 'Dataset';
    const label = item.display_name || `REFERENCE MRI ${String(index + 1).padStart(2, '0')}`;
    return `<article class="similar-card"><img src="${item.image_url}" alt="${escapeHTML(item.category || 'Same class')} reference MRI" loading="lazy"><div class="similar-card-body"><div class="similar-card-title"><span>${escapeHTML(label)}</span><small>SAME CLASS</small></div><div class="similar-card-meta">${escapeHTML(String(item.category || '').toUpperCase())} · ${escapeHTML(split)} DATASET</div><button class="secondary-button" data-open-reference="${escapeHTML(item.id)}">COMPARE WITH CURRENT SCAN ↗</button><button class="secondary-button" data-save-reference="${escapeHTML(item.id)}">${isSaved ? '★ SAVED TO COLLECTION' : '☆ SAVE REFERENCE CASE'}</button></div></article>`;
  }).join('');
  $$('[data-open-reference]').forEach(button => button.addEventListener('click', () => {
    const item = items.find(match => match.id === button.dataset.openReference);
    if (item) openComparison(item);
  }));
  $$('[data-save-reference]').forEach(button => button.addEventListener('click', () => {
    const item = items.find(match => match.id === button.dataset.saveReference);
    if (!item) return;
    const index = state.referenceFavorites.findIndex(saved => saved.id === item.id && saved.category === item.category);
    if (index >= 0) state.referenceFavorites.splice(index, 1);
    else state.referenceFavorites.push(item);
    try { localStorage.setItem('tscannerReferenceFavorites', JSON.stringify(state.referenceFavorites)); } catch (_) {}
    $('#viewSavedReferences').textContent = `☆ SAVED REFERENCES (${state.referenceFavorites.length})`;
    renderSimilarCards(items, source);
    toast(index >= 0 ? 'Reference removed from saved collection.' : 'Reference added to your saved collection.');
  }));
}
$('#viewSavedReferences').addEventListener('click', () => {
  $('#viewSavedReferences').classList.toggle('active');
  if (!state.referenceFavorites.length) {
    $('#similarGrid').innerHTML = '<div class="comparison-empty"><div class="empty-orbit">☆</div><strong>NO SAVED REFERENCE CASES</strong><p>Run a reference search and save cases to build a comparison collection on this device.</p></div>';
    return;
  }
  renderSimilarCards(state.referenceFavorites, 'saved');
});
$('#viewSavedReferences').textContent = `☆ SAVED REFERENCES (${state.referenceFavorites.length})`;

function openComparison(item) {
  const current = $('#compareCurrentImage'), reference = $('#compareReferenceImage');
  current.src = state.objectUrl || '';
  reference.src = item.image_url;
  $('#compareReferenceCaption').textContent = `${item.display_name || item.id} · ${String(item.category || '').toUpperCase()} · ${item.dataset_split || 'DATASET'} EXAMPLE`;
  const renderDiff = () => {
    if (!current.complete || !reference.complete || !current.naturalWidth || !reference.naturalWidth) return;
    const canvas = $('#compareDiffCanvas'), context = canvas.getContext('2d');
    const width = 512, height = 512; canvas.width = width; canvas.height = height;
    const a = document.createElement('canvas'), b = document.createElement('canvas'); a.width = b.width = width; a.height = b.height = height;
    const ac = a.getContext('2d'), bc = b.getContext('2d');
    ac.fillStyle = bc.fillStyle = '#05090f'; ac.fillRect(0,0,width,height); bc.fillRect(0,0,width,height);
    ac.drawImage(current,0,0,width,height); bc.drawImage(reference,0,0,width,height);
    const ap = ac.getImageData(0,0,width,height), bp = bc.getImageData(0,0,width,height), out = context.createImageData(width,height);
    for (let i=0;i<ap.data.length;i+=4) { const diff=(Math.abs(ap.data[i]-bp.data[i])+Math.abs(ap.data[i+1]-bp.data[i+1])+Math.abs(ap.data[i+2]-bp.data[i+2]))/3; const v=Math.min(255,diff*2.2); out.data[i]=v; out.data[i+1]=Math.min(255,v*.85); out.data[i+2]=Math.min(255,v*.55); out.data[i+3]=255; }
    context.putImageData(out,0,0);
  };
  current.onload = renderDiff; reference.onload = renderDiff;
  if (current.complete && reference.complete) renderDiff();
  $('#comparisonWorkbench').classList.remove('hidden');
  $('#comparisonWorkbench').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
$('#closeWorkbench').addEventListener('click', () => $('#comparisonWorkbench').classList.add('hidden'));

// ---------------------------------------------------------------------------
// Research history, model information, notes, favourites and exports.
// ---------------------------------------------------------------------------
async function loadResearch() {
  try {
    const [stats, historyData, info] = await Promise.all([api('/api/research/stats'), api(`/api/history?limit=300${state.favoritesOnly ? '&favorites_only=true' : ''}`), api('/api/model')]);
    const average = stats.average_confidence == null ? '—' : `${(stats.average_confidence * 100).toFixed(1)}%`;
    $('#researchStats').innerHTML = `<div class="research-stat"><small>ARCHIVED SCANS</small><b>${stats.total_scans}</b><span>LOCAL DATABASE</span></div><div class="research-stat"><small>TUMOUR-FLAGGED</small><b>${stats.tumor_flagged}</b><span>MODEL CLASSIFICATION</span></div><div class="research-stat"><small>AVERAGE CONFIDENCE</small><b>${average}</b><span>MODEL SCORES</span></div><div class="research-stat"><small>MODEL STATUS</small><b>${info.loaded ? 'READY' : 'OFFLINE'}</b><span>${escapeHTML(info.device || 'DEVICE UNKNOWN')}</span></div>`;
    $('#historyCount').textContent = `${historyData.count} SAVED CASE${historyData.count === 1 ? '' : 'S'}`;
    state.history = historyData.results || [];
    renderHistory();
    $('#runtimeModel').textContent = `MODEL: ${info.model_path ? info.model_path.split(/[\\/]/).at(-1) : 'UNKNOWN'}`;
    $('#runtimeClasses').textContent = `CLASSES: ${Object.values(info.classes || {}).join(', ') || 'NOT AVAILABLE'}`;
    $('#runtimeBackend').textContent = 'BACKEND: ONLINE';
  } catch (error) { toast(`Could not load research archive: ${error.message}`); }
}
function renderHistory() {
  const search = $('#historySearch').value.trim().toLowerCase();
  const items = state.history.filter(item => `${item.filename} ${item.predicted_type} ${item.created_at}`.toLowerCase().includes(search));
  if (!items.length) {
    $('#historyList').innerHTML = `<div class="comparison-empty"><div class="empty-orbit">⌁</div><strong>${state.favoritesOnly ? 'NO FAVOURITE CASES' : 'NO MATCHING CASES'}</strong><p>Analysed MRI scans are stored here in the local T-Scanner archive.</p></div>`;
    return;
  }
  $('#historyList').innerHTML = items.map(item => `<button class="history-card ${state.selectedHistory?.id === item.id ? 'active' : ''}" data-history-id="${item.id}"><img class="history-thumb" src="${item.image_url}" alt=""><span class="history-meta"><b>${escapeHTML(item.filename)}</b><small>${new Date(item.created_at).toLocaleString()}</small><em>${escapeHTML(item.predicted_type || 'Unknown')} · ${item.best_confidence == null ? '—' : (item.best_confidence * 100).toFixed(1) + '%'}</em></span><span class="history-fav">${item.favorite ? '★' : '☆'}</span></button>`).join('');
  $$('[data-history-id]').forEach(button => button.addEventListener('click', () => openHistoryCase(button.dataset.historyId)));
}
$('#historySearch').addEventListener('input', renderHistory);
$('#favoritesFilter').addEventListener('click', () => { state.favoritesOnly = !state.favoritesOnly; $('#favoritesFilter').classList.toggle('active', state.favoritesOnly); loadResearch(); });
async function openHistoryCase(id) {
  try {
    const item = await api(`/api/history/${encodeURIComponent(id)}`);
    state.selectedHistory = item;
    $('#caseDetail h3').textContent = item.filename;
    $('#caseDetail > p').textContent = `${item.predicted_type || 'Unknown'} · ${item.best_confidence == null ? 'Confidence unavailable' : (item.best_confidence * 100).toFixed(1) + '% confidence'} · ${new Date(item.created_at).toLocaleString()}`;
    $('#historyPreview').src = item.image_url; $('#historyPreview').classList.remove('hidden');
    $('#caseNotes').disabled = false; $('#caseNotes').value = item.notes || ''; $('#saveCaseNotes').disabled = false; $('#toggleFavorite').disabled = false;
    $('#toggleFavorite').textContent = item.favorite ? '★ REMOVE FAVOURITE' : '☆ MARK AS FAVOURITE';
    $('#toggleFavorite').classList.toggle('active', item.favorite);
    renderHistory();
  } catch (error) { toast(error.message); }
}
$('#saveCaseNotes').addEventListener('click', async () => {
  if (!state.selectedHistory) return;
  try {
    await api(`/api/history/${encodeURIComponent(state.selectedHistory.id)}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ notes: $('#caseNotes').value, favorite: state.selectedHistory.favorite }) });
    toast('Research notes saved.'); loadResearch();
  } catch (error) { toast(error.message); }
});
$('#toggleFavorite').addEventListener('click', async () => {
  if (!state.selectedHistory) return;
  try {
    const favorite = !state.selectedHistory.favorite;
    await api(`/api/history/${encodeURIComponent(state.selectedHistory.id)}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ notes: $('#caseNotes').value, favorite }) });
    state.selectedHistory.favorite = favorite;
    $('#toggleFavorite').textContent = favorite ? '★ REMOVE FAVOURITE' : '☆ MARK AS FAVOURITE';
    $('#toggleFavorite').classList.toggle('active', favorite);
    toast(favorite ? 'Case added to favourites.' : 'Case removed from favourites.');
    loadResearch();
  } catch (error) { toast(error.message); }
});
$('#refreshResearch').addEventListener('click', () => { loadResearch(); refreshBackend(); });
function reportEscape(value) {
  return escapeHTML(value == null || value === '' ? '—' : String(value));
}
function reportDate(value = new Date()) {
  const date = value instanceof Date ? value : new Date(value);
  return Number.isNaN(date.getTime()) ? 'Date unavailable' : date.toLocaleString(undefined, {
    year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit'
  });
}
function reportCategory(value) {
  const raw = String(value || 'Unknown').trim();
  const aliases = {
    glioma: 'Glioma', meningioma: 'Meningioma', pituitary: 'Pituitary-region tumour',
    notumor: 'No tumour class', 'no tumor': 'No tumour class', 'no tumour': 'No tumour class',
    normal: 'No tumour class', unknown: 'Unspecified'
  };
  return aliases[raw.toLowerCase().replace(/[_-]+/g, ' ')] || raw.replace(/[_-]+/g, ' ').replace(/\b\w/g, char => char.toUpperCase());
}
function reportPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(Math.max(0, Math.min(1, number)) * 100).toFixed(1)}%` : 'Not available';
}
function reportProbabilityRows(items) {
  const predictions = Array.isArray(items) ? [...items] : [];
  predictions.sort((a, b) => Number(b.probability || 0) - Number(a.probability || 0));
  if (!predictions.length) return '<tr><td colspan="2" class="muted-cell">The model did not provide a class probability breakdown.</td></tr>';
  return predictions.slice(0, 4).map(item => `<tr><td>${reportEscape(reportCategory(item.class))}</td><td class="probability-cell">${reportPercent(item.probability)}</td></tr>`).join('');
}
function reportImageFigure(src, caption, extraClass = '') {
  if (!src) return '';
  return `<figure class="scan-figure ${extraClass}"><img src="${reportEscape(src)}" alt="${reportEscape(caption)}"><figcaption>${reportEscape(caption)}</figcaption></figure>`;
}
function makeReportBoxImage(image, detections, result) {
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
  const ctx = canvas.getContext('2d'); ctx.drawImage(image, 0, 0);
  const sx = canvas.width / Number(result.image_width || canvas.width);
  const sy = canvas.height / Number(result.image_height || canvas.height);
  const valid = (detections || []).filter(item => item?.bbox && Number.isFinite(Number(item.bbox.x1)) && Number.isFinite(Number(item.bbox.y1)) && Number.isFinite(Number(item.bbox.x2)) && Number.isFinite(Number(item.bbox.y2)));
  ctx.lineWidth = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) * 0.004));
  ctx.font = `bold ${Math.max(12, Math.round(Math.min(canvas.width, canvas.height) * 0.025))}px Arial`;
  valid.forEach((item, index) => {
    const b = item.bbox;
    let x1 = Number(b.x1) * sx, y1 = Number(b.y1) * sy, x2 = Number(b.x2) * sx, y2 = Number(b.y2) * sy;
    if (Math.max(Math.abs(x1), Math.abs(y1), Math.abs(x2), Math.abs(y2)) <= 1.5) { x1 *= canvas.width; x2 *= canvas.width; y1 *= canvas.height; y2 *= canvas.height; }
    x1 = Math.max(0, Math.min(canvas.width - 1, x1)); y1 = Math.max(0, Math.min(canvas.height - 1, y1));
    x2 = Math.max(x1 + 1, Math.min(canvas.width, x2)); y2 = Math.max(y1 + 1, Math.min(canvas.height, y2));
    ctx.strokeStyle = '#36f0c1'; ctx.shadowColor = '#36f0c1'; ctx.shadowBlur = Math.max(7, ctx.lineWidth * 2); ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); ctx.shadowBlur = 0;
    const tag = `REGION ${String(index + 1).padStart(2, '0')} · ${(Number(item.confidence || 0) * 100).toFixed(1)}%`;
    const tagWidth = ctx.measureText(tag).width + 16, tagHeight = Math.max(20, ctx.lineWidth * 7);
    ctx.fillStyle = 'rgba(3,18,28,.92)'; ctx.fillRect(x1, Math.max(0, y1 - tagHeight), tagWidth, tagHeight);
    ctx.fillStyle = '#baffed'; ctx.fillText(tag, x1 + 8, Math.max(tagHeight - 5, y1 - 6));
  });
  return { dataUrl: canvas.toDataURL('image/jpeg', .92), count: valid.length };
}
function makeReportTumourCrops(image, detections, result) {
  const sx = image.naturalWidth / Number(result.image_width || image.naturalWidth);
  const sy = image.naturalHeight / Number(result.image_height || image.naturalHeight);
  const items = [];
  (detections || []).forEach((item) => {
    const b = item?.bbox; if (!b) return;
    let x1 = Number(b.x1) * sx, y1 = Number(b.y1) * sy, x2 = Number(b.x2) * sx, y2 = Number(b.y2) * sy;
    if (![x1,y1,x2,y2].every(Number.isFinite)) return;
    if (Math.max(Math.abs(x1), Math.abs(y1), Math.abs(x2), Math.abs(y2)) <= 1.5) { x1 *= image.naturalWidth; x2 *= image.naturalWidth; y1 *= image.naturalHeight; y2 *= image.naturalHeight; }
    x1 = Math.max(0, Math.min(image.naturalWidth - 1, x1)); y1 = Math.max(0, Math.min(image.naturalHeight - 1, y1));
    x2 = Math.max(x1 + 1, Math.min(image.naturalWidth, x2)); y2 = Math.max(y1 + 1, Math.min(image.naturalHeight, y2));
    const pad = Math.round(Math.max(x2 - x1, y2 - y1) * .18);
    const left = Math.max(0, Math.floor(x1 - pad)), top = Math.max(0, Math.floor(y1 - pad));
    const right = Math.min(image.naturalWidth, Math.ceil(x2 + pad)), bottom = Math.min(image.naturalHeight, Math.ceil(y2 + pad));
    const crop = document.createElement('canvas'); crop.width = Math.max(1, right - left); crop.height = Math.max(1, bottom - top);
    const ctx = crop.getContext('2d'); ctx.drawImage(image, left, top, crop.width, crop.height, 0, 0, crop.width, crop.height);
    items.push({ dataUrl: crop.toDataURL('image/jpeg', .93), confidence: Number(item.confidence || 0), label: item.class || result.predicted_type || 'Candidate region' });
  });
  return items;
}
function printDocument(title, body, options = {}) {
  const popup = window.open('', '_blank', 'width=1000,height=800');
  if (!popup) return toast('Pop-up blocked. Please allow pop-ups to create the report.');
  const styles = `
    @page{size:A4;margin:17mm 16mm 18mm}
    *{box-sizing:border-box}body{margin:0;color:#263746;background:#fff;font:11px/1.58 "Segoe UI",Arial,sans-serif;-webkit-print-color-adjust:exact;print-color-adjust:exact}
    .report{max-width:900px;margin:0 auto}.report-header{display:flex;justify-content:space-between;align-items:center;gap:20px;padding:0 0 17px;border-bottom:2px solid #dce8ed}
    .brand{display:flex;align-items:center;gap:11px}.brand-mark{width:39px;height:39px;border-radius:11px;background:#0d5363;color:#fff;display:grid;place-items:center;font-weight:800;font-size:12px;letter-spacing:.05em}
    .brand-name{font-size:16px;font-weight:750;letter-spacing:.13em;color:#163c4b}.brand-sub{font-size:8px;color:#728593;letter-spacing:.17em;text-transform:uppercase;margin-top:2px}
    .report-kind{text-align:right;color:#5d7582;font-size:9px}.report-kind strong{display:block;color:#174e5d;font-size:10px;letter-spacing:.08em;text-transform:uppercase}
    h1{font-size:25px;line-height:1.2;color:#163c4b;margin:22px 0 5px;font-weight:650}h2{font-size:14px;color:#174e5d;margin:22px 0 9px;padding-bottom:6px;border-bottom:1px solid #e3ecef}
    h3{font-size:11px;color:#225566;margin:0 0 6px}.subline{margin:0;color:#6f818d;font-size:9px}.summary-band{display:grid;grid-template-columns:1.3fr .7fr;gap:12px;margin:17px 0}
    .summary-card{border:1px solid #d9e6ea;border-radius:9px;padding:13px 15px;background:#f6fafb}.summary-card.primary{background:#edf6f7;border-color:#c9dfe3}.eyebrow{font-size:8px;color:#6b818c;text-transform:uppercase;letter-spacing:.14em}.summary-value{font-size:20px;line-height:1.25;color:#174e5d;font-weight:700;margin-top:5px}.summary-note{font-size:9px;color:#637984;margin-top:4px}
    .confidence-value{font-size:25px;color:#174e5d;font-weight:750;margin-top:3px}.section-copy{color:#435a66;margin:6px 0}.scan-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:12px 0 18px}.scan-figure{margin:0;border:1px solid #dce7ea;border-radius:8px;overflow:hidden;break-inside:avoid;background:#f5f8f9}.scan-figure img{display:block;width:100%;height:240px;object-fit:contain;background:#08111a}.scan-figure figcaption{padding:7px 10px;color:#5f7480;font-size:8px;font-weight:600}
    table{width:100%;border-collapse:collapse;margin:9px 0 15px}th{background:#edf4f6;color:#355968;text-align:left;font-size:9px;padding:8px;border-bottom:1px solid #d5e2e7}td{padding:8px;border-bottom:1px solid #e5ecef;color:#344b57;font-size:9px}td.probability-cell{text-align:right;font-weight:700;color:#174e5d}.muted-cell{color:#788994;font-style:italic}
    .finding-box{border-left:3px solid #4c9aa2;background:#f3f8f9;padding:12px 14px;border-radius:0 7px 7px 0;margin:10px 0}.finding-box p{margin:0;color:#3f5662}.plain-list{margin:7px 0 12px;padding-left:18px;color:#465d68}.plain-list li{margin:4px 0}
    .small-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin:10px 0}.small-metric{padding:10px;border:1px solid #e0e9ec;border-radius:7px}.small-metric span{display:block;font-size:8px;color:#71848d;text-transform:uppercase;letter-spacing:.08em}.small-metric b{display:block;color:#254b59;font-size:12px;margin-top:3px}
    .notice{margin-top:17px;padding:11px 13px;border:1px solid #eadfc8;background:#fbf8f0;border-radius:7px;color:#66583d;font-size:9px}.report-footer{margin-top:22px;border-top:1px solid #dce7ea;padding-top:9px;display:flex;justify-content:space-between;gap:15px;color:#7b8b93;font-size:8px}
    .archive-summary{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:17px 0}.archive-summary .summary-card{padding:12px}.archive-summary .summary-value{font-size:22px}.archive-table td:nth-child(1){white-space:nowrap;color:#71828b}.archive-table td:nth-child(3){font-weight:650;color:#244f5d}.notes-cell{max-width:260px;white-space:pre-wrap;overflow-wrap:anywhere}
    .page-break{break-before:page}.avoid-break{break-inside:avoid}
    @media screen{body{background:#eaf0f2;padding:24px}.report{background:#fff;padding:34px 42px;box-shadow:0 8px 32px #132a3b18;min-height:95vh}}
    @media print{.report{max-width:none}.scan-figure img{max-height:220px}.no-print{display:none!important}}
  `;
  popup.onload = () => {
    const images = Array.from(popup.document.images);
    Promise.all(images.map(image => image.complete ? Promise.resolve() : new Promise(resolve => {
      image.onload = resolve; image.onerror = resolve;
    }))).then(() => window.setTimeout(() => { popup.focus(); popup.print(); }, 250));
  };
  popup.document.open();
  popup.document.write(`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${reportEscape(title)}</title><style>${styles}</style></head><body><main class="report">${body}</main></body></html>`);
  popup.document.close();
}
function reportHeader(kind, generatedAt) {
  return `<header class="report-header"><div class="brand"><div class="brand-mark">TS</div><div><div class="brand-name">T-SCANNER</div><div class="brand-sub">Brain MRI Exploration & Analysis</div></div></div><div class="report-kind"><strong>${reportEscape(kind)}</strong>Prepared ${reportEscape(reportDate(generatedAt))}</div></header>`;
}
function reportFooter() {
  return `<footer class="report-footer"><span>T-Scanner · Brain MRI Analysis</span><span>For research and educational use</span></footer>`;
}

$('#printObservatory').addEventListener('click', () => {
  const cases = Array.isArray(state.history) ? state.history : [];
  const generatedAt = new Date();
  const classCounts = new Map();
  cases.forEach(item => {
    const label = reportCategory(item.predicted_type);
    classCounts.set(label, (classCounts.get(label) || 0) + 1);
  });
  const flagged = cases.filter(item => item.tumor_detected).length;
  const avg = cases.length ? cases.reduce((sum, item) => sum + (Number(item.best_confidence) || 0), 0) / cases.filter(item => item.best_confidence != null).length : null;
  const averageText = Number.isFinite(avg) ? `${(avg * 100).toFixed(1)}%` : '—';
  const categories = [...classCounts.entries()].sort((a, b) => b[1] - a[1]);
  const categoryRows = categories.length
    ? categories.map(([label, count]) => `<tr><td>${reportEscape(label)}</td><td>${count}</td><td>${cases.length ? (count / cases.length * 100).toFixed(1) : '0.0'}%</td></tr>`).join('')
    : '<tr><td colspan="3" class="muted-cell">No archived analyses are available yet.</td></tr>';
  const caseRows = cases.length
    ? cases.map((item, index) => `<tr><td>Case ${String(index + 1).padStart(3, '0')}</td><td>${reportEscape(reportDate(item.created_at))}</td><td>${reportEscape(reportCategory(item.predicted_type))}</td><td>${item.best_confidence == null ? '—' : reportPercent(item.best_confidence)}</td><td class="notes-cell">${reportEscape(item.notes || '—')}</td></tr>`).join('')
    : '<tr><td colspan="5" class="muted-cell">No cases have been saved to the research archive.</td></tr>';
  const body = `${reportHeader('Research archive summary', generatedAt)}
    <h1>Research Observatory</h1><p class="subline">A readable overview of archived MRI analysis activity and model-assigned categories.</p>
    <section class="archive-summary"><div class="summary-card"><div class="eyebrow">Archived scans</div><div class="summary-value">${cases.length}</div><div class="summary-note">Cases saved locally</div></div><div class="summary-card"><div class="eyebrow">Tumour-class results</div><div class="summary-value">${flagged}</div><div class="summary-note">Model-assigned tumour categories</div></div><div class="summary-card"><div class="eyebrow">Average confidence</div><div class="summary-value">${averageText}</div><div class="summary-note">Across available confidence scores</div></div></section>
    <h2>Category distribution</h2><table><thead><tr><th>Model-assigned category</th><th>Cases</th><th>Share</th></tr></thead><tbody>${categoryRows}</tbody></table>
    <h2>Archived case history</h2><table class="archive-table"><thead><tr><th>Reference</th><th>Analysis date</th><th>Model-assigned category</th><th>Confidence</th><th>Research notes</th></tr></thead><tbody>${caseRows}</tbody></table>
    <div class="notice"><strong>Interpretation note:</strong> These totals describe the software's saved model outputs only. They are not confirmed diagnoses, population statistics, or clinical outcome data.</div>${reportFooter()}`;
  printDocument('T-Scanner Research Observatory', body);
});

$('#printReport').addEventListener('click', () => {
  if (!state.result) return;
  const result = state.result;
  const explanation = state.explanation || {};
  const generatedAt = new Date();
  const predicted = reportCategory(result.predicted_type);
  const confidence = result.best_confidence == null ? 'Not provided' : reportPercent(result.best_confidence);
  const probabilityItems = result.top_predictions || result.class_probabilities || [];
  const detections = Array.isArray(result.detections) ? result.detections : [];
  if (!state.image || !state.image.complete || !state.image.naturalWidth) return toast('The MRI image is still loading. Please try the report again.');
  const boxImage = makeReportBoxImage(state.image, detections, result);
  const tumourCrops = makeReportTumourCrops(state.image, detections, result);
  const figureParts = [reportImageFigure(boxImage.dataUrl, boxImage.count ? 'MRI image with all returned bounding boxes' : 'MRI image · no localized bounding box was returned')];
  tumourCrops.forEach((crop, index) => figureParts.push(reportImageFigure(crop.dataUrl, `Tumour region ${index + 1} · ${reportCategory(crop.label)} · ${reportPercent(crop.confidence)} model score`)));
  const imageFigures = figureParts.join('');
  const regionCopy = detections.length
    ? `<p class="section-copy">The analysis returned ${detections.length} candidate region${detections.length === 1 ? '' : 's'} for review. These marked regions are model outputs and should be checked against the MRI by a qualified professional.</p><ul class="plain-list">${detections.slice(0, 8).map(item => `<li>${reportEscape(reportCategory(item.class || 'Candidate region'))} — ${reportPercent(item.confidence)} model score</li>`).join('')}</ul>`
    : '<p class="section-copy">The analysis did not return a localized region for this scan. This does not independently confirm the absence of an abnormality.</p>';
  const precautions = Array.isArray(explanation.precautions) ? explanation.precautions.filter(Boolean).slice(0, 5) : [];
  const precautionsHtml = precautions.length ? `<ul class="plain-list">${precautions.map(item => `<li>${reportEscape(item)}</li>`).join('')}</ul>` : '';
  const summary = explanation.summary || 'No additional written interpretation was generated. The model-assigned category and confidence are shown above.';
  const confidenceNote = explanation.confidence_note ? `<p class="section-copy">${reportEscape(explanation.confidence_note)}</p>` : '';
  const body = `${reportHeader('Brain MRI analysis report', generatedAt)}
    <h1>Brain MRI Analysis Summary</h1><p class="subline">Automated image analysis result for research and educational review.</p>
    <section class="summary-band"><div class="summary-card primary"><div class="eyebrow">AI-assigned category</div><div class="summary-value">${reportEscape(predicted)}</div><div class="summary-note">This is the category selected by the current model, not a clinical diagnosis.</div></div><div class="summary-card"><div class="eyebrow">Model confidence</div><div class="confidence-value">${reportEscape(confidence)}</div><div class="summary-note">Confidence is a model score, not the probability that a patient has a tumour.</div></div></section>
    ${imageFigures ? `<h2>Bounding-box view and individual region crops</h2><p class="section-copy">The first image shows every model-returned region. Each following image is a separate crop around one returned box, including a small surrounding margin.</p><section class="scan-grid">${imageFigures}</section>` : ''}
    <h2>Classification results</h2><p class="section-copy">The following values show how the model distributed its scores across the available categories.</p><table><thead><tr><th>Category</th><th>Model score</th></tr></thead><tbody>${reportProbabilityRows(probabilityItems)}</tbody></table>
    <h2>AI-generated summary</h2><div class="finding-box"><p>${reportEscape(summary)}</p></div>${confidenceNote}
    <h2>Image review findings</h2>${regionCopy}
    ${precautionsHtml ? `<h2>Review guidance</h2>${precautionsHtml}` : ''}
    <div class="notice"><strong>Important medical notice:</strong> This report is generated by an AI research tool. It is not a diagnosis, radiology interpretation, or treatment recommendation. Model outputs may be incorrect or incomplete. A qualified radiologist or clinician must review the original scan and relevant clinical information before any medical decision is made. Do not use this report to delay urgent medical care.</div>
    ${reportFooter()}`;
  printDocument('T-Scanner Brain MRI Analysis Summary', body);
});

// ---------------------------------------------------------------------------
// Tumour information atlas and short learning challenges.
// ---------------------------------------------------------------------------
const atlasData = {
  glioma: { title: 'Glioma', text: 'Gliomas are a broad group of tumours that arise from glial cells or their precursors within the central nervous system. Their imaging appearance, grade and behaviour vary widely by subtype.', facts: [['ORIGIN', 'Glial-lineage cells'], ['COMMON MRI CONTEXT', 'T1, T2 and FLAIR signal patterns vary'], ['IMPORTANT NOTE', 'Subtype and grade require clinical diagnosis']], note: 'A model class called “glioma” is not a histological diagnosis. MRI appearance alone cannot establish tumour subtype or grade.' },
  meningioma: { title: 'Meningioma', text: 'Meningiomas arise from the meninges, the membranes that surround the brain and spinal cord. Many are extra-axial and may appear adjacent to the dura on MRI.', facts: [['ORIGIN', 'Meningeal coverings'], ['COMMON MRI CONTEXT', 'Often extra-axial; enhancement patterns vary'], ['IMPORTANT NOTE', 'Location and pathology matter']], note: 'Imaging features can overlap with other lesions. Diagnosis and grading require appropriate specialist assessment.' },
  pituitary: { title: 'Pituitary-region tumours', text: 'The pituitary gland sits in the sella turcica at the skull base. Lesions in this region can affect nearby structures, including the optic apparatus and cavernous sinus.', facts: [['LOCATION', 'Sellar / parasellar region'], ['COMMON MRI CONTEXT', 'Dedicated pituitary MRI may be used'], ['IMPORTANT NOTE', 'Hormonal and visual assessment may be relevant']], note: 'The application class label “pituitary” is broad and does not identify a specific pituitary tumour type.' },
  notumor: { title: 'No tumour class', text: 'A “no tumour” model output means the input was assigned to the model’s non-tumour class. It does not prove that the scan is normal or exclude other neurological conditions.', facts: [['MODEL OUTPUT', 'Non-tumour class'], ['MRI CONTEXT', 'Normal appearances depend on sequences and quality'], ['IMPORTANT NOTE', 'A negative model result is not a clinical clearance']], note: 'Image quality, sequence selection and model limitations can affect output. Clinical interpretation remains essential.' }
};
function renderAtlas(key) {
  $$('#atlasTabs button').forEach(button => button.classList.toggle('active', button.dataset.atlas === key));
  if (key === 'quiz') { renderQuiz(); return; }
  const item = atlasData[key] || atlasData.glioma;
  $('#atlasContent').innerHTML = `<span class="micro-label">EDUCATIONAL REFERENCE / ${key.toUpperCase()}</span><h3>${item.title}</h3><p>${item.text}</p><div class="atlas-grid">${item.facts.map(([label, value]) => `<div class="atlas-fact"><small>${label}</small><b>${value}</b></div>`).join('')}</div><p style="margin-top:22px;color:#d2b984;font-size:10px">${item.note}</p>`;
}
function renderQuiz() {
  const questions = [
    { q: 'Which brain structure is strongly involved in balance and movement coordination?', answers: ['Cerebellum', 'Occipital lobe', 'Pituitary gland'], correct: 0, detail: 'The cerebellum contributes to balance, coordination and motor learning.' },
    { q: 'Which lobe is primarily associated with processing visual information?', answers: ['Temporal lobe', 'Occipital lobe', 'Frontal lobe'], correct: 1, detail: 'The occipital lobes contain the main cortical visual-processing areas.' },
    { q: 'Where do meningiomas typically originate?', answers: ['Meninges', 'Pituitary gland', 'Cerebellar cortex'], correct: 0, detail: 'Meningiomas arise from cells associated with the meninges.' },
    { q: 'Does a “no tumour” AI classification rule out disease?', answers: ['Yes, always', 'Only in axial scans', 'No, it is not a diagnosis'], correct: 2, detail: 'A model class is not a clinical diagnosis and cannot rule out all conditions.' }
  ];
  const item = questions[state.quizIndex % questions.length];
  $('#atlasContent').innerHTML = `<span class="micro-label">INTERACTIVE LEARNING / QUESTION ${state.quizIndex + 1}</span><h3>MRI & anatomy challenge</h3><p class="atlas-question">${item.q}</p><div class="quiz-choices">${item.answers.map((answer, index) => `<button data-quiz-answer="${index}">${answer}</button>`).join('')}</div><p id="quizFeedback" class="quiz-feedback"></p><p class="help-note">LEARNING PROGRESS · ${state.quizProgress.correct} CORRECT / ${state.quizProgress.answered} ANSWERED</p><button id="nextQuiz" class="launch-button" style="margin-top:20px"><span class="launch-icon">↻</span><span>NEXT QUESTION</span><b>↗</b></button>`;
  $$('[data-quiz-answer]').forEach(button => button.addEventListener('click', () => {
    const correct = Number(button.dataset.quizAnswer) === item.correct;
    state.quizProgress.answered += 1; if (correct) state.quizProgress.correct += 1;
    try { localStorage.setItem('tscannerQuizProgress', JSON.stringify(state.quizProgress)); } catch (_) {}
    $('#quizFeedback').textContent = `${correct ? 'Correct. ' : 'Not quite. '}${item.detail}`;
    $('#quizFeedback').style.color = correct ? '#92e7bb' : '#f0c58b';
    $$('[data-quiz-answer]').forEach(choice => { choice.disabled = true; if (Number(choice.dataset.quizAnswer) === item.correct) choice.style.borderColor = 'rgba(116,230,177,.6)'; });
  }));
  $('#nextQuiz').addEventListener('click', () => { state.quizIndex = (state.quizIndex + 1) % questions.length; renderQuiz(); });
}
$$('[data-atlas]').forEach(button => button.addEventListener('click', () => renderAtlas(button.dataset.atlas)));

// Responsive redraw when panels become visible or browser dimensions change.
window.addEventListener('resize', () => { if (state.image) drawMRI(); });

// Give the landing environment subtle interaction cues and initialise archive status.
loadResearchSafely();
async function loadResearchSafely() { try { await refreshBackend(); } catch (_) {} }
