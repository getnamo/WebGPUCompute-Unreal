// Army example: the battlefield. A flow field per goal tells every agent the shortest way around the
// obstacles (the GPU's take on "A* for every unit": one search per goal, shared by the whole army), plus the
// floor picture the agents are drawn over. Hot reloads like army_behavior.wgsl.

#include "army_common.wgsl"

// Extra cost of walking through a cell, per agent in it: higher spreads the army over more routes.
const CONGESTION: u32 = 2u;

// Each frame, first: clears the crowd counts and draws the floor (obstacles, goals, distance contours)
@compute @workgroup_size(16, 16)
fn prepare(@builtin(global_invocation_id) id: vec3u) {
	if (id.x >= DISPLAY || id.y >= DISPLAY) {
		return;
	}
	let flat = id.y * DISPLAY + id.x;
	if (flat < TEAMS * GRID * GRID) {
		atomicStore(&density[flat], 0u);
	}

	let cell = vec2i(id.xy / (DISPLAY / GRID));
	var color = vec3f(0.10, 0.11, 0.09);
	if (blocked(cell)) {
		color = vec3f(0.02, 0.02, 0.025);
	} else {
		// Contours of the distance to the east goal: they bend around whatever is in the way
		let steps = f32(distance_to(1u, cell)) / f32(STRAIGHT);
		if (steps < 100000.0 && fract(steps / 8.0) < 0.07) {
			color += vec3f(0.035, 0.045, 0.03);
		}
		let pulse = 0.3 + 0.08 * sin(params.time * 3.0);
		if (in_goal(0u, cell)) {
			color = mix(color, vec3f(0.45, 0.16, 0.08), pulse);
		}
		if (in_goal(1u, cell)) {
			color = mix(color, vec3f(0.08, 0.2, 0.45), pulse);
		}
	}
	textureStore(display, vec2i(id.xy), vec4f(color, 1.0));
}

// Then: counts the agents in each cell, per team
@compute @workgroup_size(256)
fn count(@builtin(global_invocation_id) id: vec3u) {
	if (id.x >= arrayLength(&agents)) {
		return;
	}
	let agent = agents[id.x];
	let cell = vec2i(agent.pos);
	if (agent.state != 0u && in_grid(cell)) {
		atomicAdd(&density[agent.team * GRID * GRID + cell_index(cell)], 1u);
	}
}

// Then, many times per frame: one relaxation step of both goals' distance fields (z = goal). A cell's distance
// is its cheapest neighbor's plus the step, recomputed from scratch each time, so the field also follows
// obstacles that move or appear.
@compute @workgroup_size(8, 8, 1)
fn relax(@builtin(global_invocation_id) id: vec3u) {
	if (id.x >= GRID || id.y >= GRID || id.z >= TEAMS) {
		return;
	}
	let cell = vec2i(id.xy);
	let goal = id.z;
	let index = goal * GRID * GRID + cell_index(cell);
	if (blocked(cell)) {
		atomicStore(&field[index], UNREACHABLE);
		return;
	}
	if (in_goal(goal, cell)) {
		atomicStore(&field[index], 0u);
		return;
	}

	var best = UNREACHABLE;
	for (var dy = -1; dy <= 1; dy++) {
		for (var dx = -1; dx <= 1; dx++) {
			let step = vec2i(dx, dy);
			if (all(step == vec2i(0)) || blocked(cell + step)) {
				continue;
			}
			let diagonal = dx != 0 && dy != 0;
			// No squeezing diagonally between two blocked cells
			if (diagonal && (blocked(cell + vec2i(dx, 0)) || blocked(cell + vec2i(0, dy)))) {
				continue;
			}
			best = min(best, distance_to(goal, cell + step) + select(STRAIGHT, DIAGONAL, diagonal));
		}
	}
	atomicStore(&field[index], min(best + crowd(cell) * CONGESTION, UNREACHABLE));
}
