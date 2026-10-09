import * as THREE from 'three';
import { smooth } from './kit.js';

/** One physical world and one dolly from the surface site to the chapter's orbital framing. */
export function createSurfacePullback(monolith, planet) {
  const radius = 6000, center = new THREE.Vector3(0, -radius - 3, -62);
  // The site sits at the terminator. Its local sun is low and to the left, as in the hero.
  const normal = new THREE.Vector3(-.45, .1, .887).normalize();
  const right = new THREE.Vector3(1, 0, 0).addScaledVector(normal, -normal.x).normalize();
  const back = new THREE.Vector3().crossVectors(right, normal).normalize();
  const rotation = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(right, normal, back)).invert();
  const globe = new THREE.Group(); globe.name = 'SurfacePlanet';
  globe.position.copy(center); globe.quaternion.copy(rotation); globe.scale.setScalar(radius / planet.radius);
  const bodies = planet.body.map(mesh => mesh.clone()); globe.add(...bodies);
  monolith.scene.add(globe);
  // Space replaces the low atmosphere behind the SAME terrain, without masking the foreground.
  const space = planet.sky.clone(); space.name = 'OrbitalSky'; space.renderOrder = -9;
  space.material = planet.sky.material.clone(); space.material.transparent = true;
  space.material.uniforms.uSurfaceFade = { value: 0 };
  space.material.fragmentShader = 'uniform float uSurfaceFade;\n' + space.material.fragmentShader.replace('vec4(col,1.)', 'vec4(col,uSurfaceFade)');
  monolith.scene.add(space);
  const endPosition = new THREE.Vector3(), endTarget = new THREE.Vector3(), startTarget = new THREE.Vector3();
  const localSun = new THREE.Vector3(), endUp = new THREE.Vector3(), surfaceUp = new THREE.Vector3(0, 1, 0);
  const { lands, sky, rays, mists, spindrift, dustPts, sunDir, landLight } = monolith.surface;

  function reset() {
    globe.visible = space.visible = false;
    monolith.camera.up.copy(surfaceUp);
    sky.material.uniforms.uArcFade.value = 1;
    sky.visible = rays.visible = dustPts.visible = true;
    for (const m of [...mists, ...spindrift]) m.visible = true;
    for (const land of lands) { land.visible = true; land.material.uniforms.uBend.value = 0; land.material.uniforms.uAir.value = 1; land.material.uniforms.uTerrainFade.value = 1; }
  }
  function update(progress) {
    const camera = monolith.camera;
    globe.visible = space.visible = true; globe.updateMatrixWorld(true);
    endPosition.copy(planet.camera.position).applyMatrix4(globe.matrixWorld);
    endTarget.copy(planet.target).applyMatrix4(globe.matrixWorld);
    startTarget.set(camera.position.x * .2, camera.position.y + Math.tan(THREE.MathUtils.degToRad(12)) * 100, camera.position.z - 100);
    // A straight retreat, logarithmic in distance, gives powers-of-ten scale without a second approach.
    const retreat = Math.expm1(progress * 4.8) / Math.expm1(4.8);
    camera.position.lerp(endPosition, retreat);
    startTarget.lerp(endTarget, retreat);
    camera.fov += (planet.camera.fov - camera.fov) * progress;
    endUp.copy(planet.camera.up).applyQuaternion(rotation);
    camera.up.lerpVectors(surfaceUp, endUp, smooth(0, 1, progress)).normalize();
    camera.far = 1000000; camera.updateProjectionMatrix(); camera.lookAt(startTarget); camera.updateMatrixWorld(true);
    const altitude = camera.position.distanceTo(center) - radius;
    const air = Math.exp(-Math.max(0, altitude - 2) / 180);
    space.material.uniforms.uSurfaceFade.value = smooth(5, 1200, altitude);
    space.material.uniforms.uTint.value.copy(planet.sky.material.uniforms.uTint.value);
    space.material.uniforms.uSun.value.copy(planet.sky.material.uniforms.uSun.value);
    space.material.uniforms.uAspect.value = planet.sky.material.uniforms.uAspect.value;
    space.material.uniforms.uTime.value = planet.sky.material.uniforms.uTime.value;
    sky.material.uniforms.uArcFade.value = 1 - smooth(0, .16, progress);
    sky.visible = altitude < 1200; rays.visible = altitude < 40; dustPts.visible = altitude < 20;
    for (const m of [...mists, ...spindrift]) m.visible = altitude < 350;
    for (const land of lands) {
      land.material.uniforms.uBend.value = smooth(0, .22, progress);
      land.material.uniforms.uAir.value = air;
      land.material.uniforms.uTerrainFade.value = land.name === 'Plain' ? 1 - smooth(.12, .32, progress) : 1 - smooth(.10, .30, progress);
    }
    localSun.copy(bodies[0].material.uniforms.uSun.value).applyQuaternion(rotation);
    sunDir.lerp(localSun, smooth(0, .22, progress)).normalize(); landLight.copy(sunDir);
    bodies[0].rotation.copy(planet.body[0].rotation);
    bodies[0].material.uniforms.uGround.value = 1 - smooth(800, 3000, altitude);
    bodies[1].material.uniforms.uAtmoFade.value = smooth(80, 300, altitude);
    planet.setView(camera, globe.matrixWorld);
  }
  reset();
  return { update, reset, globe, radius, center, dispose() { space.material.dispose(); } };
}
