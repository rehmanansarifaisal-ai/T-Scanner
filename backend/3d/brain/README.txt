T-SCANNER 3D BRAIN MODEL ASSET LOCATION

Copy your model into this exact structure:

backend/3d/brain/source/brain.glb
backend/3d/brain/textures/gltf_embedded_0.png
backend/3d/brain/textures/gltf_embedded_1.png
backend/3d/brain/textures/gltf_embedded_1@channels=A.png
backend/3d/brain/textures/gltf_embedded_2.png

Texture order:
- gltf_embedded_0.png: base colour
- gltf_embedded_1.png: metallic
- gltf_embedded_1@channels=A.png: roughness
- gltf_embedded_2.png: tangent-space normal map

The homepage loads the GLB directly in a native WebGL renderer. It does not use Three.js, React, Node.js or external JavaScript 3D packages.
