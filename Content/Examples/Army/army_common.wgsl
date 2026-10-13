// Army example: declarations shared by army_field.wgsl and army_behavior.wgsl (both #include this file).
// Saving it reloads every kernel of the example.
//
// The battlefield is a GRID x GRID grid over the floor of BP_WebGPUArmy. Its ObstacleCapture looks straight
// down at the same square and renders scene depth into RT_ArmyObstacles, so any mesh standing on the floor
// blocks the cells under it: move the blocks around (also while playing) and the army reroutes.

const GRID: u32 = 256u;            // cells per side (RT_ArmyObstacles is GRID x GRID)
const DISPLAY: u32 = 1024u;        // RT_ArmyDisplay size: DISPLAY / GRID pixels per cell
const TEAMS: u32 = 2u;
const GOAL_WIDTH: i32 = 6;         // cells at each end of the field that count as arrived
const UNREACHABLE: u32 = 0x3fffffffu;
const STRAIGHT: u32 = 16u;         // flow field cost of a step to a side neighbor
const DIAGONAL: u32 = 23u;         // ... and to a diagonal neighbor (16 * sqrt 2)

// The ObstacleCapture's height above the floor: the floor is seen at this depth, and anything rising
// more than BLOCK_HEIGHT above it is an obstacle.
const CAPTURE_HEIGHT: f32 = 2000.0;
const BLOCK_HEIGHT: f32 = 40.0;

// Set every frame by the Blueprint through its WebGPU Bindings (Set Float "params.time" / "params.dt")
struct Params {
	time: f32,
	dt: f32,
}

struct Agent {
	pos: vec2f,    // in cells
	vel: vec2f,    // cells per second
	team: u32,
	goal: u32,     // 0: marching to the west end (x = 0), 1: to the east end
	seed: u32,     // random state
	state: u32,    // 0 until spawned
}

// All bound by name from the Blueprint's WebGPU Bindings: each kernel only gets the ones it uses
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(2) var<storage, read_write> field: array<atomic<u32>>;    // per goal: GRID * GRID distances
@group(0) @binding(3) var<storage, read_write> density: array<atomic<u32>>;  // per team: agents in each cell
@group(0) @binding(4) var obstacles: texture_2d<f32>;                        // scene depth seen from above
@group(0) @binding(5) var display: texture_storage_2d<rgba16float, write>;   // what the floor shows

fn in_grid(cell: vec2i) -> bool {
	return all(cell >= vec2i(0)) && all(cell < vec2i(i32(GRID)));
}

fn cell_index(cell: vec2i) -> u32 {
	return u32(cell.y) * GRID + u32(cell.x);
}

fn blocked(cell: vec2i) -> bool {
	if (!in_grid(cell)) {
		return true;
	}
	let depth = textureLoad(obstacles, cell, 0).r;
	return depth > 0.0 && depth < CAPTURE_HEIGHT - BLOCK_HEIGHT;
}

fn in_goal(goal: u32, cell: vec2i) -> bool {
	// (Parentheses: `a < b, c >= d` would parse as a template list)
	return select((cell.x < GOAL_WIDTH), (cell.x >= i32(GRID) - GOAL_WIDTH), goal == 1u);
}

/** Flow field distance from a cell to a goal (UNREACHABLE outside the grid or behind walls). */
fn distance_to(goal: u32, cell: vec2i) -> u32 {
	if (!in_grid(cell)) {
		return UNREACHABLE;
	}
	return atomicLoad(&field[goal * GRID * GRID + cell_index(cell)]);
}

/** Agents of both teams in a cell. */
fn crowd(cell: vec2i) -> u32 {
	if (!in_grid(cell)) {
		return 0u;
	}
	let index = cell_index(cell);
	return atomicLoad(&density[index]) + atomicLoad(&density[GRID * GRID + index]);
}

fn hash(value: u32) -> u32 {
	// PCG
	let state = value * 747796405u + 2891336453u;
	let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

fn random(seed: ptr<function, u32>) -> f32 {
	*seed = hash(*seed);
	return f32(*seed) / 4294967295.0;
}
