import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createRoot } from 'react-dom/client';

// --- TYPE DEFINITIONS ---
type Point = { x: number; y: number };
type Road = { start: Point; end: Point };
type Q = { highway?: boolean; severed?: boolean };
type Bounds = { x: number; y: number; width: number; height: number };
type QuadtreeObject = Bounds & { o: Segment };

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
    HIGHWAY_SEGMENT_WIDTH: 80,
    DEFAULT_SEGMENT_WIDTH: 40,
    DEFAULT_SEGMENT_LENGTH: 500,
    HIGHWAY_SEGMENT_LENGTH: 500,
    MINIMUM_INTERSECTION_DEVIATION: 30,
    ROAD_SNAP_DISTANCE: 200,
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
    generate: (previousSegment: Segment, config: typeof defaultConfig, heatmap: any) => {
        const newBranches: Segment[] = [];
        if (previousSegment.q.severed) {
            return newBranches;
        }

        const template = (direction: number, length: number, t: number, q: Q) => segmentFactory.usingDirection(previousSegment.r.end, direction, length, t, q);
        const continueStraight = template(previousSegment.dir(), previousSegment.length(), 0, previousSegment.q);
        const straightPop = heatmap.popOnRoad(continueStraight.r);

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
            if (roadPop > config.HIGHWAY_BRANCH_POPULATION_THRESHOLD && Math.random() < config.HIGHWAY_BRANCH_PROBABILITY) {
                const angle = previousSegment.dir() + (Math.random() > 0.5 ? 90 : -90) + config.RANDOM_BRANCH_ANGLE();
                newBranches.push(template(angle, previousSegment.length(), 0, { highway: true }));
            }
        } else if (straightPop > config.NORMAL_BRANCH_POPULATION_THRESHOLD) {
            newBranches.push(continueStraight);
        }

        if (straightPop > config.NORMAL_BRANCH_POPULATION_THRESHOLD && Math.random() < config.DEFAULT_BRANCH_PROBABILITY) {
            const angle = previousSegment.dir() + (Math.random() > 0.5 ? 90 : -90) + config.RANDOM_BRANCH_ANGLE();
            const branchTime = previousSegment.q.highway ? config.NORMAL_BRANCH_TIME_DELAY_FROM_HIGHWAY : 0;
            newBranches.push(template(angle, config.DEFAULT_SEGMENT_LENGTH, branchTime, {}));
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
            return Math.pow((v1 * v2 + v3) / 2, 2);
        },
        popOnRoad: (r: Road) => (heatmap.populationAt(r.start.x, r.start.y) + heatmap.populationAt(r.end.x, r.end.y)) / 2
    };

    const rootSegment = new Segment({ x: 0, y: 0 }, { x: config.HIGHWAY_SEGMENT_LENGTH, y: 0 }, 0, { highway: true });
    const oppositeDirection = segmentFactory.fromExisting(rootSegment);
    oppositeDirection.setEnd({ x: -config.HIGHWAY_SEGMENT_LENGTH, y: 0 });
    oppositeDirection.links.b.push(rootSegment);
    rootSegment.links.b.push(oppositeDirection);
    priorityQ.push(rootSegment, oppositeDirection);

    while (priorityQ.length > 0 && segments.length < config.SEGMENT_COUNT_LIMIT) {
        priorityQ.sort((a, b) => a.t - b.t);
        const minSegment = priorityQ.shift()!;
        
        const accepted = localConstraints(minSegment, segments, qTree, config);
        if (accepted) {
            if (minSegment.setupBranchLinks) minSegment.setupBranchLinks();
            addSegment(minSegment, segments, qTree);
            const newBranches = globalGoals.generate(minSegment, config, heatmap);
            newBranches.forEach(branch => {
                branch.t = minSegment.t + 1 + branch.t;
                priorityQ.push(branch);
            });
        }
    }
    return { segments };
}


// --- CHARACTER LOGIC ---
const useCharacter = (segments: Segment[]) => {
    const characterState = useRef({
        position: null as Point | null,
        currentSegment: null as Segment | null,
        t: 0, // progress along the segment (0 to 1)
        speed: 150, // units per second
        image: null as HTMLImageElement | null,
        rotation: 0,
    });

    useEffect(() => {
        const image = new Image();
        image.src = 'https://raw.githubusercontent.com/eerkek/personal-code-assistant-app/main/assets/car.png';
        image.onload = () => {
            characterState.current.image = image;
        };
    }, []);

    useEffect(() => {
        if (segments.length > 0) {
            const nonHighways = segments.filter(s => !s.q.highway);
            const startSegments = nonHighways.length > 0 ? nonHighways : segments;
            const randomSegment = startSegments[Math.floor(Math.random() * startSegments.length)];

            characterState.current.currentSegment = randomSegment;
            characterState.current.position = { ...randomSegment.r.start };
            characterState.current.t = 0;
        }
    }, [segments]);

    const chooseNextSegment = (current: Segment) => {
        const connected = [...current.links.f];
        if (connected.length === 0) return null;

        const currentDir = current.dir();

        connected.sort((a, b) => {
            const aDiff = minDegreeDifference(currentDir, a.dir());
            const bDiff = minDegreeDifference(currentDir, b.dir());
            return aDiff - bDiff;
        });

        return connected[0];
    };

    const updateCharacter = (deltaTime: number) => {
        const state = characterState.current;
        if (!state.currentSegment || !state.position || !state.image) {
            return;
        }

        const segment = state.currentSegment;
        const segmentLength = segment.length();
        if (segmentLength === 0) {
             const nextSegment = chooseNextSegment(segment);
             state.currentSegment = nextSegment;
             state.t = 0;
             return;
        }

        const distanceToTravel = state.speed * deltaTime;
        state.t += distanceToTravel / segmentLength;

        if (state.t >= 1) {
            const nextSegment = chooseNextSegment(segment);
            if (nextSegment) {
                const leftoverT = state.t - 1;
                const leftoverDist = leftoverT * segmentLength;
                state.currentSegment = nextSegment;
                const nextLength = nextSegment.length();
                state.t = nextLength > 0 ? leftoverDist / nextLength : 0;
            } else {
                 const randomSegment = segments[Math.floor(Math.random() * segments.length)];
                 state.currentSegment = randomSegment;
                 if(randomSegment) state.position = { ...randomSegment.r.start };
                 state.t = 0;
                 return;
            }
        }

        const newSegment = state.currentSegment;
        if (!newSegment) return;

        const start = newSegment.r.start;
        const end = newSegment.r.end;
        state.position = {
            x: start.x + (end.x - start.x) * state.t,
            y: start.y + (end.y - start.y) * state.t,
        };
        state.rotation = newSegment.dir();
    };

    const drawCharacter = (ctx: CanvasRenderingContext2D, transform: { x: number, y: number, scale: number }) => {
        const state = characterState.current;
        if (!state.position || !state.image) return;

        const carSize = 64 / transform.scale;

        ctx.save();
        ctx.translate(state.position.x, state.position.y);
        ctx.rotate(state.rotation * Math.PI / 180);
        ctx.drawImage(state.image, -carSize / 2, -carSize / 2, carSize, carSize);
        ctx.restore();
    };

    return { updateCharacter, drawCharacter };
};


// --- REACT COMPONENT ---

const App: React.FC = () => {
    const [seed, setSeed] = useState<string>('city');
    const [segments, setSegments] = useState<Segment[]>([]);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });
    
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const transformRef = useRef({ x: 0, y: 0, scale: 1 });
    const isPanningRef = useRef(false);
    const lastMousePosRef = useRef<Point>({ x: 0, y: 0 });
    const animationFrameId = useRef<number | null>(null);
    const lastTimestamp = useRef(0);

    const { updateCharacter, drawCharacter } = useCharacter(segments);

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

        const normalRoads = segments.filter(s => !s.q.highway);
        const highways = segments.filter(s => s.q.highway);

        // Draw normal roads
        ctx.strokeStyle = 'var(--road-color)';
        ctx.lineWidth = 1.5 / scale;
        ctx.beginPath();
        normalRoads.forEach(seg => {
            ctx.moveTo(seg.r.start.x, seg.r.start.y);
            ctx.lineTo(seg.r.end.x, seg.r.end.y);
        });
        ctx.stroke();

        // Draw highways
        ctx.strokeStyle = 'var(--highway-color)';
        ctx.lineWidth = 4 / scale;
        ctx.beginPath();
        highways.forEach(seg => {
            ctx.moveTo(seg.r.start.x, seg.r.start.y);
            ctx.lineTo(seg.r.end.x, seg.r.end.y);
        });
        ctx.stroke();
        
        drawCharacter(ctx, transformRef.current);

        ctx.restore();
    }, [segments, canvasSize, drawCharacter]);

    const gameLoop = useCallback((timestamp: number) => {
        const deltaTime = (timestamp - lastTimestamp.current) / 1000;
        lastTimestamp.current = timestamp;

        updateCharacter(deltaTime);
        draw();

        animationFrameId.current = requestAnimationFrame(gameLoop);
    }, [draw, updateCharacter]);

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

        if (segments.length === 0) {
            transformRef.current = { x: canvasWidth / 2, y: canvasHeight / 2, scale: 1 };
            draw();
            return;
        }

        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const seg of segments) {
            minX = Math.min(minX, seg.r.start.x, seg.r.end.x);
            minY = Math.min(minY, seg.r.start.y, seg.r.end.y);
            maxX = Math.max(maxX, seg.r.start.x, seg.r.end.x);
            maxY = Math.max(maxY, seg.r.start.y, seg.r.end.y);
        }

        const networkWidth = (maxX - minX) || 1;
        const networkHeight = (maxY - minY) || 1;
        const networkCenterX = minX + networkWidth / 2;
        const networkCenterY = minY + networkHeight / 2;

        const padding = 0.9;
        const scale = Math.min(canvasWidth / networkWidth, canvasHeight / networkHeight) * padding;
        
        const tx = canvasWidth / 2 - networkCenterX * scale;
        const ty = canvasHeight / 2 - networkCenterY * scale;
        
        transformRef.current = { scale, x: tx, y: ty };
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
        setTimeout(() => {
            const currentSeed = seed || Date.now().toString();
            const result = generate(currentSeed);
            setSegments(result.segments);
            setIsLoading(false);
        }, 50);
    };

    const onMouseDown = (e: React.MouseEvent) => {
        isPanningRef.current = true;
        lastMousePosRef.current = { x: e.clientX, y: e.clientY };
    };

    const onMouseUp = () => {
        isPanningRef.current = false;
    };

    const onMouseMove = (e: React.MouseEvent) => {
        if (!isPanningRef.current) return;
        const dx = e.clientX - lastMousePosRef.current.x;
        const dy = e.clientY - lastMousePosRef.current.y;
        transformRef.current.x += dx;
        transformRef.current.y += dy;
        lastMousePosRef.current = { x: e.clientX, y: e.clientY };
        draw();
    };
    
    const onWheel = (e: React.WheelEvent) => {
        e.preventDefault();
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const zoomFactor = 1.1;
        const newScale = e.deltaY < 0 ? transformRef.current.scale * zoomFactor : transformRef.current.scale / zoomFactor;
        
        const worldX = (mouseX - transformRef.current.x) / transformRef.current.scale;
        const worldY = (mouseY - transformRef.current.y) / transformRef.current.scale;
        
        transformRef.current.scale = newScale;
        transformRef.current.x = mouseX - worldX * newScale;
        transformRef.current.y = mouseY - worldY * newScale;
        
        draw();
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
            </div>
            <div className="canvas-container">
                {isLoading && (
                    <div className="loader-overlay">
                        <div className="loader"></div>
                    </div>
                )}
                <canvas
                    ref={canvasRef}
                    onMouseDown={onMouseDown}
                    onMouseUp={onMouseUp}
                    onMouseLeave={onMouseUp}
                    onMouseMove={onMouseMove}
                    onWheel={onWheel}
                />
            </div>
        </div>
    );
};

const container = document.getElementById('root');
if(container) {
    const root = createRoot(container);
    root.render(<App />);
}
