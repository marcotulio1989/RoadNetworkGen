import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';
import { generateCityData, CityData } from './city-generator';

import * as THREE from 'three';
import * as turf from '@turf/turf';

function turfPolygonToThreeShape(polygon: turf.helpers.Polygon): THREE.Shape {
    const shape = new THREE.Shape();

    // Exterior ring
    const exterior = polygon.coordinates[0];
    shape.moveTo(exterior[0][0], exterior[0][1]);
    for (let i = 1; i < exterior.length; i++) {
        shape.lineTo(exterior[i][0], exterior[i][1]);
    }

    // Interior rings (holes)
    for (let i = 1; i < polygon.coordinates.length; i++) {
        const holeCoords = polygon.coordinates[i];
        const holePath = new THREE.Path();
        holePath.moveTo(holeCoords[0][0], holeCoords[0][1]);
        for (let j = 1; j < holeCoords.length; j++) {
            holePath.lineTo(holeCoords[j][0], holeCoords[j][1]);
        }
        shape.holes.push(holePath);
    }
    return shape;
}


const Cityscape: React.FC<{ data: CityData }> = ({ data }) => {
    const { boundary, roads, walks, greens, buildings } = data;

    const center = useMemo(() => {
        const bbox = turf.bbox(boundary);
        return [-(bbox[0] + bbox[2]) / 2, -(bbox[1] + bbox[3]) / 2];
    }, [boundary]);

    return (
        <group position={[center[0], 0, center[1]]} rotation={[-Math.PI / 2, 0, 0]}>
            {/* Render Buildings */}
            {buildings.map((b, i) => {
                if (!b.footprint.geometry) return null;
                const shape = turfPolygonToThreeShape(b.footprint.geometry);
                const extrudeSettings = {
                    steps: 1,
                    depth: b.height,
                    bevelEnabled: false,
                };
                const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
                return (
                    <mesh key={`bldg-${i}`} geometry={geometry} castShadow receiveShadow>
                        <meshStandardMaterial color="#cccccc" />
                    </mesh>
                );
            })}

            {/* Render Flat Features */}
            {[
                { geom: greens, color: '#88cc88' },
                { geom: walks, color: '#bbbbbb' },
                { geom: roads, color: '#444444' },
            ].map(({ geom, color }, i) => {
                if (!geom.geometry) return null;
                const polygons = geom.geometry.type === 'Polygon'
                    ? [geom.geometry]
                    : geom.geometry.coordinates.map(p => ({ type: 'Polygon', coordinates: p }));

                return polygons.map((p, j) => {
                    const shape = turfPolygonToThreeShape(p);
                    const geometry = new THREE.ShapeGeometry(shape);
                    return (
                        <mesh key={`flat-${i}-${j}`} geometry={geometry} receiveShadow>
                            <meshStandardMaterial color={color} />
                        </mesh>
                    );
                });
            })}
        </group>
    );
};

const CityCanvas: React.FC = () => {
  // useMemo will ensure the city is only generated once
  const cityData = useMemo(() => generateCityData(), []);

  return (
    <Canvas
      camera={{ position: [200, 200, 200], fov: 50 }}
      style={{ background: '#f0f0f0' }}
    >
      <ambientLight intensity={0.8} />
      <directionalLight
        position={[100, 200, 150]}
        intensity={1.5}
        castShadow
      />
      <Environment preset="city" />

      <Cityscape data={cityData} />

      <OrbitControls />
    </Canvas>
  );
};

export default CityCanvas;
