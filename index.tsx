import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createRoot } from 'react-dom/client';

// --- TYPE DEFINITIONS ---
type Point = { x: number; y: number };
type Road = { start: Point; end: Point };
type Q = { highway?: boolean; severed?: boolean };
type Bounds = { x: number; y: number; width: number; height: number };
type QuadtreeObject = Bounds & { o: Segment };
type Building = {
    footprint: Point[]; // The 4 corners of the rectangular base in world coordinates
    height: number;
};

// --- DEPENDENCY IMPLEMENTATIONS ---

/**
 * A simple seeded pseudo-random number generator.
 */
class SeededRandom {
    private seed: number;
    constructor(seedStr: string) {
        let h = 1779033703 ^ seedStr.length;
        for (let i = 0; i < seedStr.length; i++) {
            h = Math.imul(h ^ seedStr.charCodeAt(i), 3432918353);
            h = h << 13 | h >>> 19;
        }
        this.seed = this.hash(h);
    }

    private hash(h: number) {
        h = Math.imul(h ^ h >>> 16, 2246822507);
        h = Math.imul(h ^ h >>> 13, 3266489909);
        return (h ^= h >>> 16) >>> 0;
    }

    random() {
        this.seed += 1831565813;
        let t = Math.imul(this.seed ^ this.seed >>> 15, this.seed | 1);
        t = t + Math.imul(t ^ t >>> 7, t | 61);
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
}

/**
 * Simplex noise implementation.
 */
const noise = (function() {
    const F2 = 0.5 * (Math.sqrt(3.0) - 1.0);
    const G2 = (3.0 - Math.sqrt(3.0)) / 6.0;
    const grad3 = [
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
        [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]
    ];
    let p: number[] = [];
    let perm: number[] = [];
    let permMod12: number[] = [];

    function seed(random: () => number) {
        p = Array.from({ length: 256 }, (_, i) => i);
        for (let i = 255; i > 0; i--) {
            const j = Math.floor(random() * (i + 1));
            [p[i], p[j]] = [p[j], p[i]];
        }
        perm = p.concat(p);
        permMod12 = perm.map(v => v % 12);
    }

    function simplex2(xin: number, yin: number) {
        let n0, n1, n2;
        const s = (xin + yin) * F2;
        const i = Math.floor(xin + s);
        const j = Math.floor(yin + s);
        const t = (i + j) * G2;
        const X0 = i - t;
        const Y0 = j - t;
        const x0 = xin - X0;
        const y0 = yin - Y0;
        let i1, j1;
        if (x0 > y0) { i1 = 1; j1 = 0; } else { i1 = 0; j1 = 1; }
        const x1 = x0 - i1 + G2;
        const y1 = y0 - j1 + G2;
        const x2 = x0 - 1.0 + 2.0 * G2;
        const y2 = y0 - 1.0 + 2.0 * G2;
        const ii = i & 255;
        const jj = j & 255;
        let t0 = 0.5 - x0 * x0 - y0 * y0;
        if (t0 < 0) n0 = 0.0;
        else {
            t0 *= t0;
            const g = grad3[permMod12[ii + perm[jj]]];
            n0 = t0 * t0 * (g[0] * x0 + g[1] * y0);
        }
        let t1 = 0.5 - x1 * x1 - y1 * y1;
        if (t1 < 0) n1 = 0.0;
        else {
            t1 *= t1;
            const g = grad3[permMod12[ii + i1 + perm[jj + j1]]];
            n1 = t1 * t1 * (g[0] * x1 + g[1] * y1);
        }
        let t2 = 0.5 - x2 * x2 - y2 * y2;
        if (t2 < 0) n2 = 0.0;
        else {
            t2 *= t2;
            const g = grad3[permMod12[ii + 1 + perm[jj + 1]]];
            n2 = t2 * t2 * (g[0] * x2 + g[1] * y2);
        }
        return 70.0 * (n0 + n1 + n2);
    }
    return { seed, simplex2 };
})();

/**
 * Quadtree implementation for spatial partitioning.
 */
class Quadtree {
    private maxObjects: number;
    private maxLevels: number;
    private level: number;
    private bounds: Bounds;
    private objects: QuadtreeObject[] = [];
    private nodes: (Quadtree | null)[] = [];

    constructor(bounds: Bounds, maxObjects = 10, maxLevels = 10, level = 0) {
        this.bounds = bounds;
        this.maxObjects = maxObjects;
        this.maxLevels = maxLevels;
        this.level = level;
    }

    private split() {
        const nextLevel = this.level + 1;
        const subWidth = this.bounds.width / 2;
        const subHeight = this.bounds.height / 2;
        const x = this.bounds.x;
        const y = this.bounds.y;

        this.nodes[0] = new Quadtree({ x: x + subWidth, y: y, width: subWidth, height: subHeight }, this.maxObjects, this.maxLevels, nextLevel);
        this.nodes[1] = new Quadtree({ x: x, y: y, width: subWidth, height: subHeight }, this.maxObjects, this.maxLevels, nextLevel);
        this.nodes[2] = new Quadtree({ x: x, y: y + subHeight, width: subWidth, height: subHeight }, this.maxObjects, this.maxLevels, nextLevel);
        this.nodes[3] = new Quadtree({ x: x + subWidth, y: y + subHeight, width: subWidth, height: subHeight }, this.maxObjects, this.maxLevels, nextLevel);
    }

    private getIndex(pRect: Bounds): number {
        const midX = this.bounds.x + this.bounds.width / 2;
        const midY = this.bounds.y + this.bounds.height / 2;
        const top = pRect.y < midY && pRect.y + pRect.height < midY;
        const bottom = pRect.y > midY;
        if (pRect.x < midX && pRect.x + pRect.width < midX) {
            if (top) return 1;
            if (bottom) return 2;
        } else if (pRect.x > midX) {
            if (top) return 0;
            if (bottom) return 3;
        }
        return -1;
    }

    insert(pRect: QuadtreeObject) {
        if (this.nodes[0]) {
            const index = this.getIndex(pRect);
            if (index !== -1) {
                this.nodes[index]!.insert(pRect);
                return;
            }
        }
        this.objects.push(pRect);
        if (this.objects.length > this.maxObjects && this.level < this.maxLevels) {
            if (!this.nodes[0]) {
                this.split();
            }
            let i = 0;
            while (i < this.objects.length) {
                const index = this.getIndex(this.objects[i]);
                if (index !== -1) {
                    this.nodes[index]!.insert(this.objects.splice(i, 1)[0]);
                } else {
                    i++;
                }
            }
        }
    }

    retrieve(pRect: Bounds): QuadtreeObject[] {
        let returnObjects = this.objects;
        if (this.nodes[0]) {
            const index = this.getIndex(pRect);
            if (index !== -1) {
                returnObjects = returnObjects.concat(this.nodes[index]!.retrieve(pRect));
            } else {
                for (let i = 0; i < this.nodes.length; i++) {
                    const node = this.nodes[i];
                    if (node) {
                        returnObjects = returnObjects.concat(node.retrieve(pRect));
                    }
                }
            }
        }
        return returnObjects;
    }
}

/**
 * Collection of math utility functions.
 */
const math = {
    subtractPoints: (a: Point, b: Point): Point => ({ x: a.x - b.x, y: a.y - b.y }),
    length: (a: Point, b: Point): number => Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2)),
    equalV: (a: Point, b: Point): boolean => a.x === b.x && a.y === b.y,
    cosDegrees: (deg: number): number => Math.cos(deg * Math.PI / 180),
    sinDegrees: (deg: number): number => Math.sin(deg * Math.PI / 180),
    toDegrees: (rad: number): number => rad * 180 / Math.PI,
    toIsometric: (p: Point): Point => {
        return {
            x: p.x - p.y,
            y: (p.x + p.y) * 0.45,
        };
    },
    fromIsometric: (p: Point): Point => {
        return {
            x: p.y / 0.9 + p.x / 2,
            y: p.y / 0.9 - p.x / 2,
        };
    },
    doLineSegmentsIntersect: (p: Point, p2: Point, q: Point, q2: Point, touchIsIntersect = false): { t: number, point: Point } | null => {
        const r = { x: p2.x - p.x, y: p2.y - p.y };
        const s = { x: q2.x - q.x, y: q2.y - q.y };
        const rxs = r.x * s.y - r.y * s.x;
        const qpxr = (q.x - p.x) * r.y - (q.y - p.y) * r.x;
        if (rxs === 0 && qpxr === 0) return null; // Collinear
        if (rxs === 0 && qpxr !== 0) return null; // Parallel and non-intersecting
        const t = ((q.x - p.x) * s.y - (q.y - p.y) * s.x) / rxs;
        const u = ((q.x - p.x) * r.y - (q.y - p.y) * r.x) / rxs;
        const check = touchIsIntersect ? (v: number) => v >= 0 && v <= 1 : (v: number) => v > 0 && v < 1;
        if (check(t) && check(u)) {
            return { t, point: { x: p.x + t * r.x, y: p.y + t * r.y } };
        }
        return null;
    },
    distanceToLine: (point: Point, start: Point, end: Point) => {
        const l2 = Math.pow(start.x - end.x, 2) + Math.pow(start.y - end.y, 2);
        if (l2 === 0) return { distance2: Math.pow(point.x - start.x, 2) + Math.pow(point.y - start.y, 2), lineProj2: 0, length2: 0, pointOnLine: start };
        let t = ((point.x - start.x) * (end.x - start.x) + (point.y - start.y) * (end.y - start.y)) / l2;
        t = Math.max(0, Math.min(1, t));
        const pointOnLine = { x: start.x + t * (end.x - start.x), y: start.y + t * (end.y - start.y) };
        return {
            distance2: Math.pow(point.x - pointOnLine.x, 2) + Math.pow(point.y - pointOnLine.y, 2),
            lineProj2: t,
            length2: l2,
            pointOnLine
        };
    }
};

// --- ROAD GENERATION LOGIC ---

const defaultConfig = {
    HIGHWAY_SEGMENT_WIDTH: 640,
    DEFAULT_SEGMENT_WIDTH: 320,
    DEFAULT_SEGMENT_LENGTH: 4000,
    HIGHWAY_SEGMENT_LENGTH: 4000,
    MINIMUM_INTERSECTION_DEVIATION: 30,
    ROAD_SNAP_DISTANCE: 400,
    RANDOM_STRAIGHT_ANGLE: () => (Math.random() - 0.5) * 2 * 10,
    RANDOM_BRANCH_ANGLE: () => (Math.random() - 0.5) * 2 * 20,
    HIGHWAY_BRANCH_POPULATION_THRESHOLD: 0.3,
    NORMAL_BRANCH_POPULATION_THRESHOLD: 0.1,
    HIGHWAY_BRANCH_PROBABILITY: 0.1,
    DEFAULT_BRANCH_PROBABILITY: 0.4,
    NORMAL_BRANCH_TIME_DELAY_FROM_HIGHWAY: 10,
    SEGMENT_COUNT_LIMIT: 2000,
    QUADTREE_BOUNDS: { x: -15000, y: -15000, width: 30000, height: 30000 },
    QUADTREE_MAX_OBJECTS: 10,
    QUADTREE_MAX_LEVELS: 10,
};

let segmentCounter = 0;
class Segment {
    id: number;
    r: Road;
    t: number;
    q: Q;
    links: { b: Segment[]; f: Segment[] };
    width: number;
    private roadRevision = 0;
    private dirRevision?: number;
    private lengthRevision?: number;
    private cachedDir?: number;
    private cachedLength?: number;
    setupBranchLinks?: () => void;

    constructor(start: Point, end: Point, t: number, q: Q) {
        this.id = segmentCounter++;
        this.r = { start, end };
        this.t = t || 0;
        this.q = q || {};
        this.links = { b: [], f: [] };
        this.width = this.q.highway ? defaultConfig.HIGHWAY_SEGMENT_WIDTH : defaultConfig.DEFAULT_SEGMENT_WIDTH;
    }
    
    setStart(val: Point) { this.r.start = val; this.roadRevision++; };
    setEnd(val: Point) { this.r.end = val; this.roadRevision++; };

    dir(): number {
        if (this.dirRevision !== this.roadRevision) {
            this.dirRevision = this.roadRevision;
            const vector = math.subtractPoints(this.r.end, this.r.start);
            // standard math angle: 0 is +X, 90 is +Y
            const angleRad = Math.atan2(vector.y, vector.x);
            this.cachedDir = math.toDegrees(angleRad);
        }
        return this.cachedDir!;
    }

    length(): number {
        if (this.lengthRevision !== this.roadRevision) {
            this.lengthRevision = this.roadRevision;
            this.cachedLength = math.length(this.r.start, this.r.end);
        }
        return this.cachedLength!;
    }
    
    split(point: Point, segment: Segment, segmentList: Segment[], qTree: Quadtree) {
        const startIsBackwards = this.startIsBackwards();
        const splitPart = segmentFactory.fromExisting(this);
        addSegment(splitPart, segmentList, qTree);
        splitPart.setEnd(point);
        this.setStart(point);

        splitPart.links.b = [...this.links.b];
        splitPart.links.f = [...this.links.f];

        const firstSplit = startIsBackwards ? splitPart : this;
        const secondSplit = startIsBackwards ? this : splitPart;
        const fixLinks = startIsBackwards ? splitPart.links.b : splitPart.links.f;

        fixLinks.forEach(link => {
            let index = link.links.b.indexOf(this);
            if (index !== -1) {
                link.links.b[index] = splitPart;
            } else {
                index = link.links.f.indexOf(this);
                if (index !== -1) {
                    link.links.f[index] = splitPart;
                }
            }
        });
        firstSplit.links.f = [segment, secondSplit];
        secondSplit.links.b = [segment, firstSplit];
        segment.links.f.push(firstSplit, secondSplit);
    }

    startIsBackwards() {
        if (this.links.b.length > 0) {
          return math.equalV(this.links.b[0].r.start, this.r.start) ||
                 math.equalV(this.links.b[0].r.end, this.r.start);
        } else if (this.links.f.length > 0) {
          return math.equalV(this.links.f[0].r.start, this.r.end) ||
                 math.equalV(this.links.f[0].r.end, this.r.end);
        }
        return false;
    }
    
    linksForEndContaining(segment: Segment) {
        if (this.links.b.indexOf(segment) > -1) return this.links.b;
        if (this.links.f.indexOf(segment) > -1) return this.links.f;
        return undefined;
    }
}

function minDegreeDifference(a: number, b: number): number {
    const diff = Math.abs(a - b) % 360;
    return Math.min(diff, 360 - diff);
}

const segmentFactory = {
    fromExisting: (segment: Segment) => new Segment(segment.r.start, segment.r.end, segment.t, JSON.parse(JSON.stringify(segment.q))),
    usingDirection: (start: Point, dir: number, length: number, t: number, q: Q) => {
        const end = {
            x: start.x + length * math.cosDegrees(dir),
            y: start.y + length * math.sinDegrees(dir)
        };
        return new Segment(start, end, t, q);
    }
};

function addSegment(segment: Segment, segmentList: Segment[], qTree: Quadtree) {
    segmentList.push(segment);
    const limits = {
        x: Math.min(segment.r.start.x, segment.r.end.x) - segment.width,
        y: Math.min(segment.r.start.y, segment.r.end.y) - segment.width,
        width: Math.abs(segment.r.start.x - segment.r.end.x) + 2 * segment.width,
        height: Math.abs(segment.r.start.y - segment.r.end.y) + 2 * segment.width,
    };
    qTree.insert({ ...limits, o: segment });
}

function localConstraints(segment: Segment, segments: Segment[], qTree: Quadtree, config: typeof defaultConfig) {
    let action = { priority: 0, func: undefined as (() => boolean) | undefined, q: { t: 0 } };
    const matches = qTree.retrieve({
        x: Math.min(segment.r.start.x, segment.r.end.x) - config.ROAD_SNAP_DISTANCE,
        y: Math.min(segment.r.start.y, segment.r.end.y) - config.ROAD_SNAP_DISTANCE,
        width: Math.abs(segment.r.start.x - segment.r.end.x) + 2 * config.ROAD_SNAP_DISTANCE,
        height: Math.abs(segment.r.start.y - segment.r.end.y) + 2 * config.ROAD_SNAP_DISTANCE,
    });

    for (const match of matches) {
        const other = match.o;
        if (other === segment) continue;

        if (action.priority <= 4) {
            const intersection = math.doLineSegmentsIntersect(segment.r.start, segment.r.end, other.r.start, other.r.end);
            if (intersection) {
                if (!action.q.t || intersection.t < action.q.t) {
                    action.q.t = intersection.t;
                    action.priority = 4;
                    action.func = () => {
                        if (minDegreeDifference(other.dir(), segment.dir()) < config.MINIMUM_INTERSECTION_DEVIATION) return false;
                        other.split(intersection.point, segment, segments, qTree);
                        segment.setEnd(intersection.point);
                        segment.q.severed = true;
                        return true;
                    };
                }
            }
        }

        if (action.priority <= 3) {
            if (math.length(segment.r.end, other.r.end) <= config.ROAD_SNAP_DISTANCE) {
                action.priority = 3;
                action.func = () => {
                    segment.setEnd(other.r.end);
                    segment.q.severed = true;
                    return true;
                }
            }
        }
        if (action.priority <= 2) {
            const dist = math.distanceToLine(segment.r.end, other.r.start, other.r.end);
            if (dist.distance2 < config.ROAD_SNAP_DISTANCE * config.ROAD_SNAP_DISTANCE && dist.lineProj2 >= 0 && dist.lineProj2 <= 1) {
                action.priority = 2;
                action.func = () => {
                    if (minDegreeDifference(other.dir(), segment.dir()) < config.MINIMUM_INTERSECTION_DEVIATION) return false;
                    segment.setEnd(dist.pointOnLine);
                    segment.q.severed = true;
                    other.split(dist.pointOnLine, segment, segments, qTree);
                    return true;
                }
            }
        }
    }

    if (action.func) return action.func();
    return true;
}

const globalGoals = {
    generate: (previousSegment: Segment, config: typeof defaultConfig, heatmap: any, log: (message: string) => void) => {
        const newBranches: Segment[] = [];
        if (previousSegment.q.severed) {
            return newBranches;
        }

        const template = (direction: number, length: number, t: number, q: Q) => segmentFactory.usingDirection(previousSegment.r.end, direction, length, t, q);
        const continueStraight = template(previousSegment.dir(), previousSegment.length(), 0, previousSegment.q);
        const straightPop = heatmap.popOnRoad(continueStraight.r);
        log(`  - Straight pop: ${straightPop.toFixed(3)} (Threshold: ${config.NORMAL_BRANCH_POPULATION_THRESHOLD})`);

        if (previousSegment.q.highway) {
            const randomStraight = template(previousSegment.dir() + config.RANDOM_STRAIGHT_ANGLE(), previousSegment.length(), 0, previousSegment.q);
            const randomPop = heatmap.popOnRoad(randomStraight.r);
            let roadPop;
            if (randomPop > straightPop) {
                newBranches.push(randomStraight);
                roadPop = randomPop;
            } else {
                newBranches.push(continueStraight);
                roadPop = straightPop;
            }
            log(`  - Highway road pop: ${roadPop.toFixed(3)} (Threshold: ${config.HIGHWAY_BRANCH_POPULATION_THRESHOLD})`);
            if (roadPop > config.HIGHWAY_BRANCH_POPULATION_THRESHOLD && Math.random() < config.HIGHWAY_BRANCH_PROBABILITY) {
                const angle = previousSegment.dir() + (Math.random() > 0.5 ? 90 : -90) + config.RANDOM_BRANCH_ANGLE();
                newBranches.push(template(angle, previousSegment.length(), 0, { highway: true }));
                log(`  - SUCCESS: Highway branching condition met.`);
            }
        } else if (straightPop > config.NORMAL_BRANCH_POPULATION_THRESHOLD) {
            newBranches.push(continueStraight);
            log(`  - SUCCESS: Normal road continuation condition met.`);
        }

        if (straightPop > config.NORMAL_BRANCH_POPULATION_THRESHOLD && Math.random() < config.DEFAULT_BRANCH_PROBABILITY) {
            const angle = previousSegment.dir() + (Math.random() > 0.5 ? 90 : -90) + config.RANDOM_BRANCH_ANGLE();
            const branchTime = previousSegment.q.highway ? config.NORMAL_BRANCH_TIME_DELAY_FROM_HIGHWAY : 0;
            newBranches.push(template(angle, config.DEFAULT_SEGMENT_LENGTH, branchTime, {}));
            log(`  - SUCCESS: Default branching condition met.`);
        }

        newBranches.forEach(branch => {
            branch.setupBranchLinks = () => {
                previousSegment.links.f.forEach(link => {
                    branch.links.b.push(link);
                    const links = link.linksForEndContaining(previousSegment);
                    if (links) {
                        links.push(branch);
                    }
                });
                previousSegment.links.f.push(branch);
                branch.links.b.push(previousSegment);
            };
        });
        return newBranches;
    }
};

function generate(seed: string, options: Partial<typeof defaultConfig> = {}) {
    const logs: string[] = [];
    const log = (message: string) => {
        logs.push(message);
    };

    segmentCounter = 0;
    const config = { ...defaultConfig, ...options };
    const random = new SeededRandom(seed);
    Math.random = random.random.bind(random);
    noise.seed(Math.random);

    const segments: Segment[] = [];
    const priorityQ: Segment[] = [];
    const qTree = new Quadtree(config.QUADTREE_BOUNDS, config.QUADTREE_MAX_OBJECTS, config.QUADTREE_MAX_LEVELS);
    
    const heatmap = {
        populationAt: (x: number, y: number) => {
            const v1 = (noise.simplex2(x / 10000, y / 10000) + 1) / 2;
            const v2 = (noise.simplex2(x / 20000 + 500, y / 20000 + 500) + 1) / 2;
            const v3 = (noise.simplex2(x / 20000 + 1000, y / 20000 + 1000) + 1) / 2;
            return Math.pow((v1 + v2 + v3) / 3, 2);
        },
        popOnRoad: (r: Road) => (heatmap.populationAt(r.start.x, r.start.y) + heatmap.populationAt(r.end.x, r.end.y)) / 2
    };

    const rootSegment = new Segment({ x: 0, y: 0 }, { x: config.HIGHWAY_SEGMENT_LENGTH, y: 0 }, 0, { highway: true });
    const oppositeDirection = segmentFactory.fromExisting(rootSegment);
    oppositeDirection.setEnd({ x: -config.HIGHWAY_SEGMENT_LENGTH, y: 0 });
    oppositeDirection.links.b.push(rootSegment);
    rootSegment.links.b.push(oppositeDirection);
    priorityQ.push(rootSegment, oppositeDirection);

    log("Starting road generation...");

    while (priorityQ.length > 0 && segments.length < config.SEGMENT_COUNT_LIMIT) {
        priorityQ.sort((a, b) => a.t - b.t);
        const minSegment = priorityQ.shift()!;
        
        log(`---\nProcessing segment ${minSegment.id}. Queue: ${priorityQ.length}, Segments: ${segments.length}`);

        const accepted = localConstraints(minSegment, segments, qTree, config);
        if (accepted) {
            if (minSegment.setupBranchLinks) minSegment.setupBranchLinks();
            addSegment(minSegment, segments, qTree);

            log(`Segment ${minSegment.id} accepted. Generating branches...`);
            const newBranches = globalGoals.generate(minSegment, config, heatmap, log);

            if (newBranches.length > 0) {
                log(`-> Created ${newBranches.length} new branches.`);
            } else {
                log(`-> No branches created for segment ${minSegment.id}.`);
            }

            newBranches.forEach(branch => {
                branch.t = minSegment.t + 1 + branch.t;
                priorityQ.push(branch);
            });
        } else {
            log(`Segment ${minSegment.id} rejected by local constraints.`);
        }
    }
    log("Finished road generation.");
    return { segments, logs };
}

function findCityBlocks(segments: Segment[]): Point[][] {
    if (segments.length === 0) {
        return [];
    }

    type Junction = {
        point: Point;
        segments: Segment[];
    };

    const junctions = new Map<string, Junction>();
    const pointToKey = (p: Point) => `${p.x},${p.y}`;

    // 1. Build junctions map
    for (const seg of segments) {
        const startKey = pointToKey(seg.r.start);
        const endKey = pointToKey(seg.r.end);

        if (!junctions.has(startKey)) {
            junctions.set(startKey, { point: seg.r.start, segments: [] });
        }
        if (!junctions.has(endKey)) {
            junctions.set(endKey, { point: seg.r.end, segments: [] });
        }

        junctions.get(startKey)!.segments.push(seg);
        junctions.get(endKey)!.segments.push(seg);
    }

    // 2. Sort segments at each junction by angle
    for (const junction of junctions.values()) {
        junction.segments.sort((a, b) => {
            const getOtherEnd = (seg: Segment, point: Point) => math.equalV(seg.r.start, point) ? seg.r.end : seg.r.start;
            const angleA = Math.atan2(
                getOtherEnd(a, junction.point).y - junction.point.y,
                getOtherEnd(a, junction.point).x - junction.point.x
            );
            const angleB = Math.atan2(
                getOtherEnd(b, junction.point).y - junction.point.y,
                getOtherEnd(b, junction.point).x - junction.point.x
            );
            return angleA - angleB;
        });
    }

    const faces: Point[][] = [];
    const visitedHalfEdges = new Set<string>(); // Key: segment.id + "," + startPointKey

    // 3. Face-finding traversal
    for (const startJunction of junctions.values()) {
        for (const startSegment of startJunction.segments) {
            const startKey = pointToKey(startJunction.point);
            const halfEdgeKey = `${startSegment.id},${startKey}`;

            if (visitedHalfEdges.has(halfEdgeKey)) {
                continue;
            }

            const newFace: Point[] = [];
            let currentJunction = startJunction;
            let currentSegment = startSegment;
            let pathFound = false;

            for (let i = 0; i < segments.length + 1; i++) { // Loop breaker
                const currentKey = pointToKey(currentJunction.point);
                const currentHalfEdgeKey = `${currentSegment.id},${currentKey}`;

                if (visitedHalfEdges.has(currentHalfEdgeKey)) {
                    break;
                }
                visitedHalfEdges.add(currentHalfEdgeKey);
                newFace.push(currentJunction.point);

                const nextPoint = math.equalV(currentSegment.r.start, currentJunction.point)
                    ? currentSegment.r.end
                    : currentSegment.r.start;

                const nextJunction = junctions.get(pointToKey(nextPoint));
                if (!nextJunction) break;

                const sortedSegments = nextJunction.segments;
                const incomingIndex = sortedSegments.findIndex(s => s.id === currentSegment.id);

                if (incomingIndex === -1) break;

                const nextSegment = sortedSegments[(incomingIndex + 1) % sortedSegments.length];

                currentJunction = nextJunction;
                currentSegment = nextSegment;

                if (currentSegment.id === startSegment.id && math.equalV(nextPoint, startJunction.point)) {
                    pathFound = true;
                    break;
                }
            }

            if (pathFound && newFace.length > 2) {
                faces.push(newFace);
            }
        }
    }

    console.log(`Found ${faces.length} faces.`);
    return faces;
}

function generateAllBuildings(blocks: Point[][]): Building[] {
    const allBuildings: Building[] = [];
    const buildingMinSize = 150;
    const buildingMaxSize = 400;

    for (const block of blocks) {
        if (block.length < 3) continue;

        // 1. Find bounding box of the block
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        block.forEach(p => {
            minX = Math.min(minX, p.x);
            minY = Math.min(minY, p.y);
            maxX = Math.max(maxX, p.x);
            maxY = Math.max(maxY, p.y);
        });

        const blockWidth = maxX - minX;
        const blockHeight = maxY - minY;

        if (blockWidth < buildingMinSize * 1.5 || blockHeight < buildingMinSize * 1.5) {
            continue; // Skip blocks that are too small
        }

        // 2. Decide on building size
        const buildingWidth = Math.max(buildingMinSize, Math.random() * Math.min(blockWidth * 0.5, buildingMaxSize));
        const buildingDepth = Math.max(buildingMinSize, Math.random() * Math.min(blockHeight * 0.5, buildingMaxSize));

        // 3. Find a random position within the block's bounding box
        const x = minX + (blockWidth - buildingWidth) / 2;
        const y = minY + (blockHeight - buildingDepth) / 2;

        // For now, we don't check if the building is actually inside the polygon,
        // the bounding box placement is good enough for a first version.

        const footprint: Point[] = [
            { x: x, y: y },
            { x: x + buildingWidth, y: y },
            { x: x + buildingWidth, y: y + buildingDepth },
            { x: x, y: y + buildingDepth },
        ];

        allBuildings.push({
            footprint,
            height: Math.random() * 800 + 200, // Random height
        });
    }

    console.log(`Generated ${allBuildings.length} buildings.`);
    return allBuildings;
}


// --- CHARACTER LOGIC ---
const useCharacter = () => {
    const [isImageLoaded, setIsImageLoaded] = useState(false);
    const characterState = useRef({
        position: { x: 0, y: 0 } as Point,
        speed: 400, // units per second
        image: null as HTMLImageElement | null,
        rotation: 0,
    });

    const keys = useRef({
        ArrowUp: false,
        ArrowDown: false,
        ArrowLeft: false,
        ArrowRight: false,
    });

    useEffect(() => {
        const image = new Image();
        image.src = 'https://raw.githubusercontent.com/eerkek/personal-code-assistant-app/main/assets/car.png';
        image.onload = () => {
            characterState.current.image = image;
            setIsImageLoaded(true);
        };

        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key in keys.current) {
                (keys.current as any)[e.key] = true;
            }
        };

        const handleKeyUp = (e: KeyboardEvent) => {
            if (e.key in keys.current) {
                (keys.current as any)[e.key] = false;
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        window.addEventListener('keyup', handleKeyUp);

        return () => {
            window.removeEventListener('keydown', handleKeyDown);
            window.removeEventListener('keyup', handleKeyUp);
        };
    }, []);

    const updateCharacter = (deltaTime: number) => {
        const state = characterState.current;
        const keysPressed = keys.current;

        if (!state.position) {
            return;
        }

        const speed = state.speed * deltaTime;
        let screen_dx = 0;
        let screen_dy = 0;

        if (keysPressed.ArrowUp) {
            screen_dy = -1;
        }
        if (keysPressed.ArrowDown) {
            screen_dy = 1;
        }
        if (keysPressed.ArrowLeft) {
            screen_dx = -1;
        }
        if (keysPressed.ArrowRight) {
            screen_dx = 1;
        }

        if (screen_dx !== 0 || screen_dy !== 0) {
            const screen_vec = { x: screen_dx, y: screen_dy };
            const world_vec = math.fromIsometric(screen_vec);

            const len = Math.sqrt(world_vec.x * world_vec.x + world_vec.y * world_vec.y);
            if (len > 0) {
                const dx = (world_vec.x / len) * speed;
                const dy = (world_vec.y / len) * speed;

                state.position.x += dx;
                state.position.y += dy;
                state.rotation = Math.atan2(dy, dx) * 180 / Math.PI;
            }
        }
    };

    const drawCharacter = (ctx: CanvasRenderingContext2D, transform: { x: number, y: number, scale: number }) => {
        const state = characterState.current;
        if (!state.position) return;

        const carSize = 64;
        const isoPos = math.toIsometric(state.position);

        ctx.save();
        ctx.translate(isoPos.x, isoPos.y);
        ctx.rotate(state.rotation * Math.PI / 180);

        if (state.image) {
            ctx.drawImage(state.image, -carSize / 2, -carSize / 2, carSize, carSize);
        } else {
            // Fallback circle
            ctx.beginPath();
            ctx.arc(0, 0, carSize / 2, 0, 2 * Math.PI);
            ctx.fillStyle = 'blue';
            ctx.fill();
        }
        ctx.restore();
    };

    return { updateCharacter, drawCharacter, characterState };
};


// --- REACT COMPONENT ---

const App: React.FC = () => {
    const [seed, setSeed] = useState<string>('city');
    const [segments, setSegments] = useState<Segment[]>([]);
    const [blocks, setBlocks] = useState<Point[][]>([]);
    const [buildings, setBuildings] = useState<Building[]>([]);
    const [logs, setLogs] = useState<string[]>([]);
    const [charPos, setCharPos] = useState<Point>({ x: 0, y: 0 });
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });
    
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const minimapCanvasRef = useRef<HTMLCanvasElement>(null);
    const transformRef = useRef({ x: 0, y: 0, scale: 1 });
    const animationFrameId = useRef<number | null>(null);
    const lastTimestamp = useRef(0);

    const { updateCharacter, drawCharacter, characterState } = useCharacter();

    const drawMinimap = useCallback(() => {
        const minimapCanvas = minimapCanvasRef.current;
        if (!minimapCanvas || segments.length === 0) return;

        const minimapCtx = minimapCanvas.getContext('2d');
        if (!minimapCtx) return;

        const { width, height } = minimapCanvas;
        minimapCanvas.width = width;
        minimapCanvas.height = height;

        // 1. Find bounds of all roads
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        segments.forEach(seg => {
            minX = Math.min(minX, seg.r.start.x, seg.r.end.x);
            minY = Math.min(minY, seg.r.start.y, seg.r.end.y);
            maxX = Math.max(maxX, seg.r.start.x, seg.r.end.x);
            maxY = Math.max(maxY, seg.r.start.y, seg.r.end.y);
        });

        // Add some padding
        const padding = 2000;
        minX -= padding;
        minY -= padding;
        maxX += padding;
        maxY += padding;

        const mapWidth = maxX - minX;
        const mapHeight = maxY - minY;

        // 2. Calculate scale and offset
        const scaleX = width / mapWidth;
        const scaleY = height / mapHeight;
        const scale = Math.min(scaleX, scaleY);

        const offsetX = (width - mapWidth * scale) / 2 - minX * scale;
        const offsetY = (height - mapHeight * scale) / 2 - minY * scale;

        minimapCtx.clearRect(0, 0, width, height);
        minimapCtx.save();
        minimapCtx.translate(offsetX, offsetY);
        minimapCtx.scale(scale, scale);

        // 3. Draw roads
        const normalRoads = segments.filter(s => !s.q.highway);
        const highways = segments.filter(s => s.q.highway);

        const roadWidth = defaultConfig.DEFAULT_SEGMENT_WIDTH / 4;
        const highwayWidth = defaultConfig.HIGHWAY_SEGMENT_WIDTH / 4;

        minimapCtx.strokeStyle = '#888888';
        minimapCtx.lineWidth = roadWidth;
        minimapCtx.lineCap = "round";
        minimapCtx.beginPath();
        normalRoads.forEach(seg => {
            minimapCtx.moveTo(seg.r.start.x, seg.r.start.y);
            minimapCtx.lineTo(seg.r.end.x, seg.r.end.y);
        });
        minimapCtx.stroke();

        minimapCtx.strokeStyle = '#f5a623';
        minimapCtx.lineWidth = highwayWidth;
        minimapCtx.lineCap = "round";
        minimapCtx.beginPath();
        highways.forEach(seg => {
            minimapCtx.moveTo(seg.r.start.x, seg.r.start.y);
            minimapCtx.lineTo(seg.r.end.x, seg.r.end.y);
        });
        minimapCtx.stroke();

        // 4. Draw character
        const charPos = characterState.current.position;
        if (charPos) {
            minimapCtx.fillStyle = 'red';
            minimapCtx.beginPath();
            minimapCtx.arc(charPos.x, charPos.y, 150, 0, 2 * Math.PI);
            minimapCtx.fill();
        }

        // 5. Draw viewport
        const mainTransform = transformRef.current;
        const { width: mainWidth, height: mainHeight } = canvasSize;

        const viewRectWorldX = (mainWidth / 2 - mainTransform.x) / mainTransform.scale - mainWidth / (2 * mainTransform.scale);
        const viewRectWorldY = (mainHeight / 2 - mainTransform.y) / mainTransform.scale - mainHeight / (2 * mainTransform.scale);
        const viewRectWorldWidth = mainWidth / mainTransform.scale;
        const viewRectWorldHeight = mainHeight / mainTransform.scale;

        minimapCtx.strokeStyle = 'black';
        minimapCtx.lineWidth = 150;
        minimapCtx.strokeRect(viewRectWorldX, viewRectWorldY, viewRectWorldWidth, viewRectWorldHeight);

        minimapCtx.restore();

    }, [segments, characterState, canvasSize]);

    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        const { width, height } = canvasSize;
        if (!canvas || width === 0 || height === 0) return;
        
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        canvas.width = width;
        canvas.height = height;

        const { x, y, scale } = transformRef.current;
        
        ctx.clearRect(0, 0, width, height);
        ctx.save();
        ctx.translate(x, y);
        ctx.scale(scale, scale);

        // --- NEW SIMPLIFIED RENDERING LOGIC ---

        const drawRoads = () => {
            const normalRoads = segments.filter(s => !s.q.highway);
            const highways = segments.filter(s => s.q.highway);

            const drawRoadsAsPolygons = (roads: Segment[], color: string) => {
                ctx.fillStyle = color;
                roads.forEach(seg => {
                    const vec = math.subtractPoints(seg.r.end, seg.r.start);
                    const length = Math.sqrt(vec.x * vec.x + vec.y * vec.y);
                    if (length === 0) return;
                    const perp = { x: -vec.y / length, y: vec.x / length };
                    const halfWidth = seg.width / 2;
                    const p1 = { x: seg.r.start.x + perp.x * halfWidth, y: seg.r.start.y + perp.y * halfWidth };
                    const p2 = { x: seg.r.start.x - perp.x * halfWidth, y: seg.r.start.y - perp.y * halfWidth };
                    const p3 = { x: seg.r.end.x - perp.x * halfWidth, y: seg.r.end.y - perp.y * halfWidth };
                    const p4 = { x: seg.r.end.x + perp.x * halfWidth, y: seg.r.end.y + perp.y * halfWidth };
                    const iso_p1 = math.toIsometric(p1);
                    const iso_p2 = math.toIsometric(p2);
                    const iso_p3 = math.toIsometric(p3);
                    const iso_p4 = math.toIsometric(p4);
                    ctx.beginPath();
                    ctx.moveTo(iso_p1.x, iso_p1.y);
                    ctx.lineTo(iso_p2.x, iso_p2.y);
                    ctx.lineTo(iso_p3.x, iso_p3.y);
                    ctx.lineTo(iso_p4.x, iso_p4.y);
                    ctx.closePath();
                    ctx.fill();
                });
            };

            drawRoadsAsPolygons(highways, 'var(--highway-color)');
            drawRoadsAsPolygons(normalRoads, 'var(--road-color)');
        };

        const drawBuildingsAndCharacter = () => {
            type Renderable = {
                zIndex: number;
                type: 'building' | 'character';
                data: Building | { position: Point };
            };
            const renderables: Renderable[] = [];

            buildings.forEach(b => {
                const centerX = b.footprint.reduce((sum, p) => sum + p.x, 0) / 4;
                const centerY = b.footprint.reduce((sum, p) => sum + p.y, 0) / 4;
                renderables.push({ zIndex: centerX + centerY, type: 'building', data: b });
            });

            if (characterState.current.position) {
                const charPos = characterState.current.position;
                renderables.push({ zIndex: charPos.x + charPos.y, type: 'character', data: { position: charPos } });
            }

            renderables.sort((a, b) => a.zIndex - b.zIndex);

            renderables.forEach(r => {
                if (r.type === 'building') {
                    const building = r.data as Building;
                    const { footprint, height } = building;
                    const base_iso = footprint.map(math.toIsometric);
                    const top_iso = footprint.map(p => {
                        const iso = math.toIsometric(p);
                        iso.y -= height;
                        return iso;
                    });
                    ctx.strokeStyle = '#222222';
                    ctx.lineWidth = 2;
                    ctx.fillStyle = '#555555';
                    ctx.beginPath();
                    ctx.moveTo(base_iso[1].x, base_iso[1].y);
                    ctx.lineTo(base_iso[2].x, base_iso[2].y);
                    ctx.lineTo(top_iso[2].x, top_iso[2].y);
                    ctx.lineTo(top_iso[1].x, top_iso[1].y);
                    ctx.closePath();
                    ctx.fill();
                    ctx.stroke();
                    ctx.fillStyle = '#666666';
                    ctx.beginPath();
                    ctx.moveTo(base_iso[2].x, base_iso[2].y);
                    ctx.lineTo(base_iso[3].x, base_iso[3].y);
                    ctx.lineTo(top_iso[3].x, top_iso[3].y);
                    ctx.lineTo(top_iso[2].x, top_iso[2].y);
                    ctx.closePath();
                    ctx.fill();
                    ctx.stroke();
                    ctx.fillStyle = '#8a8a8a';
                    ctx.beginPath();
                    ctx.moveTo(top_iso[0].x, top_iso[0].y);
                    ctx.lineTo(top_iso[1].x, top_iso[1].y);
                    ctx.lineTo(top_iso[2].x, top_iso[2].y);
                    ctx.lineTo(top_iso[3].x, top_iso[3].y);
                    ctx.closePath();
                    ctx.fill();
                    ctx.stroke();
                } else if (r.type === 'character') {
                    drawCharacter(ctx, transformRef.current);
                }
            });
        };

        drawRoads();
        drawBuildingsAndCharacter();

        ctx.restore();

        drawMinimap();
    }, [segments, buildings, canvasSize, drawCharacter, drawMinimap, characterState]);

    const gameLoop = useCallback((timestamp: number) => {
        const deltaTime = (timestamp - lastTimestamp.current) / 1000;
        lastTimestamp.current = timestamp;

        updateCharacter(deltaTime);
        setCharPos({ ...characterState.current.position });

        if (characterState.current.position) {
            const { width: canvasWidth, height: canvasHeight } = canvasSize;
            const scale = transformRef.current.scale;
            const isoPos = math.toIsometric(characterState.current.position);
            transformRef.current.x = canvasWidth / 2 - isoPos.x * scale;
            transformRef.current.y = canvasHeight / 2 - isoPos.y * scale;
        }

        draw();

        animationFrameId.current = requestAnimationFrame(gameLoop);
    }, [draw, updateCharacter, canvasSize, characterState, setCharPos]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const resizeObserver = new ResizeObserver(entries => {
            if (entries[0]) {
                const { width, height } = entries[0].contentRect;
                setCanvasSize({ width, height });
            }
        });
        resizeObserver.observe(canvas);

        return () => resizeObserver.disconnect();
    }, []);
    
    useEffect(() => {
        const { width: canvasWidth, height: canvasHeight } = canvasSize;
        if (canvasWidth === 0 || canvasHeight === 0) return;

        // I am removing the fixed scale and calculating it dynamically.
        // This will fix the "stretching" issue by ensuring the view scales
        // uniformly based on the canvas dimensions.
        const smallerDimension = Math.min(canvasWidth, canvasHeight);
        // This sets the scale so that a world view of 730 units (approx. 20 meters) fits into the smaller dimension.
        transformRef.current.scale = smallerDimension / 1095;

        if (segments.length === 0) {
            transformRef.current.x = canvasWidth / 2;
            transformRef.current.y = canvasHeight / 2;
        }
        
        draw();
    }, [segments, canvasSize, draw]);

    useEffect(() => {
        lastTimestamp.current = performance.now();
        animationFrameId.current = requestAnimationFrame(gameLoop);
        return () => {
            if (animationFrameId.current) {
                cancelAnimationFrame(animationFrameId.current);
            }
        };
    }, [gameLoop]);

    const handleGenerate = () => {
        setIsLoading(true);
        setLogs([]);
        setTimeout(() => {
            const currentSeed = seed || Date.now().toString();
            const result = generate(currentSeed);
            setSegments(result.segments);
            const blocks = findCityBlocks(result.segments);
            setBlocks(blocks);
            const buildings = generateAllBuildings(blocks);
            setBuildings(buildings);

            if (buildings.length > 0 && characterState.current) {
                const b = buildings[Math.floor(buildings.length / 2)]; // Pick a building near the middle
                const centerX = b.footprint[0].x + (b.footprint[1].x - b.footprint[0].x) / 2;
                const centerY = b.footprint[0].y + (b.footprint[3].y - b.footprint[0].y) / 2;
                characterState.current.position = { x: centerX, y: centerY };
            }

            setLogs(result.logs);
            setIsLoading(false);
        }, 50);
    };

    useEffect(() => {
      handleGenerate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    return (
        <div className="app-container">
            <div className="controls">
                <h1>Road Network Generator</h1>
                <div className="input-group">
                    <label htmlFor="seed">Seed:</label>
                    <input
                        id="seed"
                        type="text"
                        value={seed}
                        onChange={(e) => setSeed(e.target.value)}
                        placeholder="Enter a seed..."
                        disabled={isLoading}
                    />
                </div>
                <button onClick={handleGenerate} disabled={isLoading}>
                    Generate
                </button>
                <div className="char-position">
                    X: {charPos.x.toFixed(0)} | Y: {charPos.y.toFixed(0)}
                </div>
            </div>
            <div className="canvas-container">
                {isLoading && (
                    <div className="loader-overlay">
                        <div className="loader"></div>
                    </div>
                )}
                <canvas
                    ref={canvasRef}
                />
                <canvas
                    ref={minimapCanvasRef}
                    className="minimap"
                />
            </div>
            {logs.length > 0 && (
                <pre className="logs-container">
                    {logs.join('\n')}
                </pre>
            )}
        </div>
    );
};

const container = document.getElementById('root');
if(container) {
    const root = createRoot(container);
    root.render(<App />);
}
