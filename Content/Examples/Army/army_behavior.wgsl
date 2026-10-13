// Army example: how every agent behaves. This is the file to play with.
//
// Edit and save it while the example plays: the kernels reload within half a second and the army carries on
// from where it is with the new rules. If an edit doesn't compile, the error shows on screen and the last
// good version keeps running. Things to try:
//   - SPEED = 40.0: a sprinting army
//   - SEPARATION = 0.0: they pile up into narrow streams; 3.0: they spread out
//   - in steer(): `return -flow(*agent) * SPEED;` sends everyone away from their goal
//   - in steer(): rotate the flow, e.g. `let f = flow(*agent); return vec2f(-f.y, f.x) * SPEED;`
//   - agent_color(): paint them by speed, by goal, by crowd
// Press R in the example to respawn everyone.

#include "army_common.wgsl"

const SPEED: f32 = 14.0;        // cells per second
const TURN: f32 = 5.0;          // how fast the velocity follows the desired direction
const SEPARATION: f32 = 0.8;    // push away from crowded cells
const WANDER: f32 = 0.3;        // random jitter, relative to SPEED

fn spawn(index: u32) -> Agent {
	var agent: Agent;
	var seed = hash(index + 1u);
	agent.team = index % TEAMS;
	// Team 0 starts at the west end and marches east, team 1 the other way
	let depth = random(&seed) * 28.0;
	let y = 6.0 + random(&seed) * f32(GRID - 12u);
	agent.seed = seed;
	agent.pos = vec2f(select(2.0 + depth, f32(GRID) - 2.0 - depth, agent.team == 1u), y);
	agent.vel = vec2f(0.0);
	agent.goal = 1u - agent.team;
	agent.state = 1u;
	return agent;
}

// Downhill on the goal's flow field: the shortest way around everything in the way
fn flow(agent: Agent) -> vec2f {
	let cell = vec2i(agent.pos);
	var best = distance_to(agent.goal, cell);
	var direction = vec2f(0.0);
	for (var dy = -1; dy <= 1; dy++) {
		for (var dx = -1; dx <= 1; dx++) {
			let distance = distance_to(agent.goal, cell + vec2i(dx, dy));
			if (distance < best) {
				best = distance;
				direction = normalize(vec2f(f32(dx), f32(dy)));
			}
		}
	}
	return direction;
}

// Away from the neighboring cells that hold the most agents
fn separation(agent: Agent) -> vec2f {
	let cell = vec2i(agent.pos);
	var push = vec2f(0.0);
	for (var dy = -1; dy <= 1; dy++) {
		for (var dx = -1; dx <= 1; dx++) {
			push -= vec2f(f32(dx), f32(dy)) * f32(crowd(cell + vec2i(dx, dy)));
		}
	}
	return push * 0.25;
}

// The velocity an agent wants this frame
fn steer(agent: ptr<function, Agent>) -> vec2f {
	var seed = (*agent).seed;
	let wander = vec2f(random(&seed) - 0.5, random(&seed) - 0.5) * WANDER;
	(*agent).seed = seed;
	return (flow(*agent) + wander) * SPEED + separation(*agent) * SEPARATION;
}

fn agent_color(agent: Agent) -> vec3f {
	let base = select(vec3f(1.0, 0.42, 0.16), vec3f(0.25, 0.62, 1.0), agent.team == 1u);
	// Brighter while marching home
	return base * select(0.75, 1.15, agent.goal == agent.team);
}

@compute @workgroup_size(256)
fn simulate(@builtin(global_invocation_id) id: vec3u) {
	let index = id.x;
	if (index >= arrayLength(&agents)) {
		return;
	}
	var agent = agents[index];
	if (agent.state == 0u) {
		agent = spawn(index);
	}

	let dt = min(params.dt, 0.05);
	let desired = steer(&agent);
	agent.vel += (desired - agent.vel) * min(TURN * dt, 1.0);

	// Move, sliding along walls instead of entering them
	var next = agent.pos + agent.vel * dt;
	if (blocked(vec2i(next))) {
		if (!blocked(vec2i(vec2f(next.x, agent.pos.y)))) {
			next = vec2f(next.x, agent.pos.y);
			agent.vel.y = 0.0;
		} else if (!blocked(vec2i(vec2f(agent.pos.x, next.y)))) {
			next = vec2f(agent.pos.x, next.y);
			agent.vel.x = 0.0;
		} else {
			next = agent.pos;
			agent.vel = vec2f(0.0);
		}
	}
	agent.pos = clamp(next, vec2f(0.5), vec2f(f32(GRID) - 0.5));

	// Arrived: turn around and march back
	if (in_goal(agent.goal, vec2i(agent.pos))) {
		agent.goal = 1u - agent.goal;
	}
	agents[index] = agent;

	// Two by two pixels on the floor
	let pixel = vec2i(agent.pos * f32(DISPLAY / GRID));
	let color = vec4f(agent_color(agent), 1.0);
	textureStore(display, pixel, color);
	textureStore(display, pixel + vec2i(1, 0), color);
	textureStore(display, pixel + vec2i(0, 1), color);
	textureStore(display, pixel + vec2i(1, 1), color);
}

// Press R: everyone back to the start
@compute @workgroup_size(256)
fn respawn(@builtin(global_invocation_id) id: vec3u) {
	if (id.x < arrayLength(&agents)) {
		agents[id.x] = spawn(id.x);
	}
}
