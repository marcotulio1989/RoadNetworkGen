import * as turf from '@turf/turf';
import KDBush from 'kdbush';

// =========================================================================
// TYPES
// =========================================================================
export interface CityData {
    boundary: turf.Feature<turf.Polygon | turf.MultiPolygon>;
    roads: turf.Feature<turf.Polygon | turf.MultiPolygon>;
    walks: turf.Feature<turf.Polygon | turf.MultiPolygon>;
    lots: turf.Feature<turf.Polygon>[];
    buildings: {
        footprint: turf.Feature<turf.Polygon>;
        height: number;
    }[];
    greens: turf.Feature<turf.Polygon | turf.MultiPolygon>;
}

// =========================================================================
// MAIN GENERATOR FUNCTION
// =========================================================================

export function generateCityData(): CityData {
    // =========================================================================
    // USER MAPS
    // =========================================================================
    const W = 1200;
    const H = 900;
    const xCoords = linspace(0, W, 120); // Use a coarser grid for performance in JS
    const yCoords = linspace(0, H, 90);
    const [X, Y] = meshgrid(xCoords, yCoords);

    // --- Population-density map ---
    const pd: number[][] = Array(yCoords.length).fill(0).map(() => Array(xCoords.length).fill(0));
    for (let r = 0; r < yCoords.length; r++) {
        for (let c = 0; c < xCoords.length; c++) {
            const x = X[r][c];
            const y = Y[r][c];
            const val =
                0.4 * Math.exp(-((x - 300) ** 2 + (y - 200) ** 2) / (2 * 120 ** 2)) +
                0.6 * Math.exp(-((x - 880) ** 2 + (y - 650) ** 2) / (2 * 180 ** 2)) +
                0.3 * Math.exp(-((x - 950) ** 2 + (y - 200) ** 2) / (2 * 120 ** 2)) +
                0.2 * Math.random() * 0.05;
            pd[r][c] = val;
        }
    }

    // --- Zoning map: 1=Res, 2=Com, 3=Ind, 4=Park ---
    const zone: number[][] = Array(yCoords.length).fill(0).map(() => Array(xCoords.length).fill(1));
    for (let r = 0; r < yCoords.length; r++) {
        for (let c = 0; c < xCoords.length; c++) {
            if (pd[r][c] > 0.45) zone[r][c] = 2; // Commercial
            else if (pd[r][c] < 0.12) zone[r][c] = 4; // Park
            else if (pd[r][c] > 0.28 && pd[r][c] <= 0.45 && X[r][c] < 750) zone[r][c] = 3; // Industrial
        }
    }

    // --- Height-limit map (simplified, no gaussian filter) ---
    const pdFlat = pd.flat();
    const pdMin = Math.min(...pdFlat);
    const pdMax = Math.max(...pdFlat);
    const hlim: number[][] = pd.map(row =>
        row.map(val => 12 + 40 * ((val - pdMin) / (pdMax - pdMin)))
    );

    // --- Forbidden/water mask ---
    const water: boolean[][] = Array(yCoords.length).fill(false).map(() => Array(xCoords.length).fill(false));
    for (let r = 0; r < yCoords.length; r++) {
        for (let c = 0; c < xCoords.length; c++) {
            const x = X[r][c];
            const y = Y[r][c];
            if ((y < 80) || (y > H - 40) || (x < 60) || (x > W - 60)) {
                water[r][c] = true;
            }
            const water_body = Math.exp(-((x - 600) ** 2 + (y - (450-90)) ** 2) / (2 * 180 ** 2)) > 0.5;
            if (water_body && y < 450) {
                water[r][c] = true;
            }
        }
    }
    const forbid = water;

    // --- City boundary and water bodies ---
    let boundary = turf.polygon([[[0, 0], [W, 0], [W, H], [0, H], [0, 0]]]);

    const waterPerimeter1 = turf.polygon([[[0, 0], [W, 0], [W, 80], [0, 80], [0, 0]]]);
    const waterPerimeter2 = turf.polygon([[[0, H - 40], [W, H - 40], [W, H], [0, H], [0, H - 40]]]);
    const waterPerimeter3 = turf.polygon([[[0, 0], [60, 0], [60, H], [0, H], [0, 0]]]);
    const waterPerimeter4 = turf.polygon([[[W - 60, 0], [W, 0], [W, H], [W - 60, H], [W - 60, 0]]]);
    const centralWater = turf.circle([600, 450-90], 180 * 0.7, { units: 'meters' }); // Simplified circle

    let water = turf.union(waterPerimeter1, waterPerimeter2);
    water = turf.union(water, waterPerimeter3);
    water = turf.union(water, waterPerimeter4);
    water = turf.union(water, centralWater);

    boundary = turf.difference(boundary, water);

    console.log("Step 1/4: Maps and boundary generated (TS).");

    // =========================================================================
    // ROADS
    // =========================================================================
    const roadOpts = {
        hwy: { step: 20, turnStd: (10 * Math.PI) / 180, len: 260, snap: 8, biasToPeak: 0.9 },
        str: { step: 10, turnStd: (18 * Math.PI) / 180, len: 120, snap: 6, gridBias: 0.65, biasToPeak: 0.35 }
    };

    // --- Highways ---
    const hwySeeds = findLocalMax(pd, X, Y, 3);
    const Hwy = growRoads(hwySeeds, pd, forbid, xCoords, yCoords, boundary, roadOpts.hwy);

    // --- Streets ---
    let streetSeeds = downsamplePolyline(Hwy, 80);
    const extraStreetSeeds = findLocalMax(pd, X, Y, 8);
    streetSeeds = streetSeeds.concat(extraStreetSeeds);
    const Str = growRoads(streetSeeds, pd, forbid, xCoords, yCoords, boundary, roadOpts.str);

    // --- Buffer roads and create sidewalks ---
    const PR_hwy = turf.buffer(Hwy, 12, { units: 'meters' });
    const PR_st = turf.buffer(Str, 6, { units: 'meters' });
    const roads = turf.union(PR_hwy, PR_st);
    const walks = turf.buffer(roads, 3, { units: 'meters' });

    console.log("Step 2/4: Road network generated (TS).");

    // =========================================================================
    // BLOCKS -> LOTS -> BUILDINGS
    // =========================================================================
    const cityLand = turf.difference(boundary, walks);
    let blocks: turf.Feature<turf.Polygon>[] = [];
    if (cityLand.geometry.type === 'Polygon') {
        blocks.push(cityLand as turf.Feature<turf.Polygon>);
    } else { // MultiPolygon
        blocks = cityLand.geometry.coordinates.map(p => turf.polygon(p));
    }
    blocks = blocks.filter(b => turf.area(b) > 2500);

    let lots: turf.Feature<turf.Polygon>[] = [];
    for (const block of blocks) {
        lots.push(...subdivLots(block, 900, 0.15));
    }

    lots = lots.filter(lot => turf.area(lot) > 120);
    lots = lots.filter(lot => turf.booleanIntersects(turf.buffer(lot, 1, {units: 'meters'}), roads));

    const buildings: CityData['buildings'] = [];
    const lotPolygons = lots.map(l => l.geometry);

    for (const lot of lots) {
        const center = turf.centroid(lot);
        const [x, y] = center.geometry.coordinates;
        const z = sampleGrid([x, y], zone, xCoords, yCoords);
        const hcap = sampleGrid([x, y], hlim, xCoords, yCoords);

        let cov = 0, hin = [0, 0];
        if (z > 0.9 && z < 1.1) { // Res
            cov = 0.45 + 0.15 * Math.random(); hin = [6, 18];
        } else if (z > 1.9 && z < 2.1) { // Com
            cov = 0.65 + 0.10 * Math.random(); hin = [12, 60];
        } else if (z > 2.9 && z < 3.1) { // Ind
            cov = 0.55 + 0.10 * Math.random(); hin = [8, 24];
        } else { // Park
            cov = 0.10 + 0.05 * Math.random(); hin = [0, 0];
        }

        const footprint = shrinkToCoverage(lot, cov);
        const height = Math.min(hin[0] + Math.random() * (hin[1] - hin[0]), hcap);
        buildings.push({ footprint, height });
    }

    const allLots = turf.featureCollection(lots);
    const allFootprints = turf.featureCollection(buildings.map(b => b.footprint));
    const greens = turf.difference(turf.combine(allLots), turf.combine(allFootprints));

    console.log("Step 3/4: Blocks, lots, and buildings generated (TS).");

    return { boundary, roads, walks, lots, buildings, greens };
}


// =========================================================================
// HELPERS
// =========================================================================

function linspace(start: number, stop: number, num: number): number[] {
    const arr: number[] = [];
    const step = (stop - start) / (num - 1);
    for (let i = 0; i < num; i++) {
        arr.push(start + (step * i));
    }
    return arr;
}

function meshgrid(x: number[], y: number[]): [number[][], number[][]] {
    const X: number[][] = [];
    const Y: number[][] = [];
    for (let i = 0; i < y.length; i++) {
        X.push(x);
        const yRow = Array(x.length).fill(y[i]);
        Y.push(yRow);
    }
    return [X, Y];
}

// More helpers for road generation

function findLocalMax(grid: number[][], X: number[][], Y: number[][], k: number): number[][] {
    const smoothed = boxBlur(grid, 1);
    const maxima: { val: number; r: number; c: number }[] = [];
    for (let r = 1; r < grid.length - 1; r++) {
        for (let c = 1; c < grid[0].length - 1; c++) {
            const val = smoothed[r][c];
            let isMax = true;
            for (let dr = -1; dr <= 1; dr++) {
                for (let dc = -1; dc <= 1; dc++) {
                    if (dr === 0 && dc === 0) continue;
                    if (smoothed[r + dr][c + dc] > val) {
                        isMax = false;
                        break;
                    }
                }
                if (!isMax) break;
            }
            if (isMax) {
                maxima.push({ val, r, c });
            }
        }
    }
    maxima.sort((a, b) => b.val - a.val);
    return maxima.slice(0, k).map(m => [X[m.r][m.c], Y[m.r][m.c]]);
}

function growRoads(
    seeds: number[][],
    densGrid: number[][],
    forbidGrid: boolean[][],
    xCoords: number[], yCoords: number[],
    boundary: turf.Feature,
    opts: any
): turf.Feature<turf.MultiLineString> {
    const lines: turf.Feature<turf.LineString>[] = [];
    const vertexIndex = new KDBush(seeds.length, 64, Float64Array);
    seeds.forEach(s => vertexIndex.add(s[0], s[1]));

    const dirs = [0, 90, 180, 270].map(d => (d * Math.PI) / 180);

    for (const seed of seeds) {
        let pos = seed;
        let ang = Math.random() * 2 * Math.PI;
        const nSteps = Math.round(opts.len / opts.step);

        for (let i = 0; i < nSteps; i++) {
            const [gx, gy] = gradAt(pos, densGrid, xCoords, yCoords);
            let goal = Math.atan2(gy, gx);
            if (!isFinite(goal)) goal = ang;

            if (opts.gridBias && Math.random() < opts.gridBias) {
                const angDiffs = dirs.map(d => Math.abs(wrapToPi(ang - d)));
                goal = dirs[angDiffs.indexOf(Math.min(...angDiffs))];
            }

            ang += opts.turnStd * (Math.random() * 2 - 1) + opts.biasToPeak * wrapToPi(goal - ang);
            const nxt = [pos[0] + opts.step * Math.cos(ang), pos[1] + opts.step * Math.sin(ang)];

            if (!turf.booleanPointInPolygon(nxt, boundary)) break;
            if (sampleGrid(nxt, forbidGrid, xCoords, yCoords) > 0.5) break;

            const neighbors = vertexIndex.within(nxt[0], nxt[1], opts.snap);
            if (neighbors.length > 0) {
                const closestIdx = neighbors[0];
                const closestVertex = [vertexIndex.points[closestIdx*2], vertexIndex.points[closestIdx*2 + 1]];
                lines.push(turf.lineString([pos, closestVertex]));
                pos = closestVertex; // Snap to the existing vertex
                break; // End this road branch
            }

            lines.push(turf.lineString([pos, nxt]));
            vertexIndex.add(nxt[0], nxt[1]);
            pos = nxt;
        }
    }
    const fc = turf.featureCollection(lines);
    if (fc.features.length === 0) return turf.multiLineString([]);
    return turf.combine(fc);
}

function downsamplePolyline(multiline: turf.Feature<turf.MultiLineString>, ds: number): number[][] {
    const points: number[][] = [];
    if (!multiline || !multiline.geometry) return points;

    for (const line of multiline.geometry.coordinates) {
        const lineString = turf.lineString(line);
        const len = turf.length(lineString, { units: 'meters' });
        for (let d = 0; d < len; d += ds) {
            const p = turf.along(lineString, d, { units: 'meters' });
            points.push(p.geometry.coordinates);
        }
    }
    return points;
}

function sampleGrid(p: number[], grid: number[][], xCoords: number[], yCoords: number[]): number {
    const [x, y] = p;
    const c = (x - xCoords[0]) / (xCoords[1] - xCoords[0]);
    const r = (y - yCoords[0]) / (yCoords[1] - yCoords[0]);
    const c0 = Math.floor(c), c1 = Math.ceil(c);
    const r0 = Math.floor(r), r1 = Math.ceil(r);
    if (r0 < 0 || r1 >= grid.length || c0 < 0 || c1 >= grid[0].length) return 0;

    const v00 = grid[r0][c0], v01 = grid[r0][c1];
    const v10 = grid[r1][c0], v11 = grid[r1][c1];
    const wx = c - c0;
    const r_v0 = v00 * (1 - wx) + v01 * wx;
    const r_v1 = v10 * (1 - wx) + v11 * wx;
    const wy = r - r0;
    return r_v0 * (1 - wy) + r_v1 * wy;
}

function gradAt(p: number[], grid: number[][], xCoords: number[], yCoords: number[]): [number, number] {
    const h = 0.1; // A small step for central difference
    const gx = (sampleGrid([p[0] + h, p[1]], grid, xCoords, yCoords) - sampleGrid([p[0] - h, p[1]], grid, xCoords, yCoords)) / (2 * h);
    const gy = (sampleGrid([p[0], p[1] + h], grid, xCoords, yCoords) - sampleGrid([p[0], p[1] - h], grid, xCoords, yCoords)) / (2 * h);
    return [gx, gy];
}

function boxBlur(grid: number[][], r: number): number[][] {
    const out = grid.map(row => [...row]);
    for (let i = r; i < grid.length - r; i++) {
        for (let j = r; j < grid[0].length - r; j++) {
            let sum = 0;
            for (let dr = -r; dr <= r; dr++) {
                for (let dc = -r; dc <= r; dc++) {
                    sum += grid[i + dr][j + dc];
                }
            }
            out[i][j] = sum / ((2 * r + 1) ** 2);
        }
    }
    return out;
}

function wrapToPi(angle: number): number {
    return (angle + Math.PI) % (2 * Math.PI) - Math.PI;
}

// Helpers for lots and buildings

function subdivLots(polygon: turf.Feature<turf.Polygon>, targetArea: number, anisotropy: number): turf.Feature<turf.Polygon>[] {
    const lots: turf.Feature<turf.Polygon>[] = [];
    const todo = [polygon];
    const MAX_ITER = 1000; // Safety break
    let iter = 0;

    while (todo.length > 0 && iter < MAX_ITER) {
        iter++;
        const p = todo.shift();
        if (!p || turf.area(p) <= 1) continue;

        if (turf.area(p) <= targetArea * (0.7 + 0.6 * Math.random())) {
            lots.push(p);
            continue;
        }

        const bbox = turf.bbox(p);
        const w = bbox[2] - bbox[0];
        const h = bbox[3] - bbox[1];
        let dir = (w > h) ? 0 : Math.PI / 2;
        dir += anisotropy * (Math.random() - 0.5) * Math.PI / 2;

        const center = turf.centroid(p).geometry.coordinates;
        const dx = Math.cos(dir);
        const dy = Math.sin(dir);

        // Create a large polygon to act as a cutting half-plane
        const largeDist = Math.max(w, h) * 2;
        const p1 = [center[0] - dx * largeDist, center[1] - dy * largeDist];
        const p2 = [center[0] + dx * largeDist, center[1] + dy * largeDist];
        const p3 = [p2[0] - dy * largeDist, p2[1] + dx * largeDist];
        const p4 = [p1[0] - dy * largeDist, p1[1] + dx * largeDist];
        const cutter = turf.polygon([[p1, p2, p3, p4, p1]]);

        try {
            const part1 = turf.intersect(p, cutter);
            if (part1) {
                // The other part is the difference
                const part2 = turf.difference(p, part1 as turf.Feature<turf.Polygon | turf.MultiPolygon>);

                if (part1.geometry.type === 'Polygon') {
                    todo.push(part1 as turf.Feature<turf.Polygon>);
                } else { // MultiPolygon
                    part1.geometry.coordinates.forEach(c => todo.push(turf.polygon(c)));
                }

                if (part2 && part2.geometry.type === 'Polygon') {
                    todo.push(part2 as turf.Feature<turf.Polygon>);
                } else if (part2) { // MultiPolygon
                    part2.geometry.coordinates.forEach(c => todo.push(turf.polygon(c)));
                }
            } else {
                // If intersect fails, don't try to split this polygon further
                lots.push(p);
            }
        } catch (e) {
            console.error("Error during polygon subdivision:", e);
            lots.push(p); // Add problematic polygon to lots to avoid infinite loop
        }
    }
    return lots;
}

function shrinkToCoverage(lot: turf.Feature<turf.Polygon>, coverage: number): turf.Feature<turf.Polygon> {
    const lotArea = turf.area(lot);
    if (lotArea <= 0) return lot;

    let d0 = 0, d1 = Math.sqrt(lotArea) / 2;

    for (let i = 0; i < 10; i++) {
        const dm = (d0 + d1) / 2;
        const shrunk = turf.buffer(lot, -dm, { units: 'meters' });
        if (!shrunk || turf.area(shrunk) / lotArea > coverage) {
            d0 = dm;
        } else {
            d1 = dm;
        }
    }
    return turf.buffer(lot, -d0, { units: 'meters' });
}
