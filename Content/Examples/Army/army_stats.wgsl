// Army example: per-team numbers for the script to show (examples/webgpu/army.js reads them back every frame).
// Counters per team t, at stats[t * 4 + ...]:
//   0: agents marching east      1: agents in the east half
//   2: summed speed, in 1/16 cells per second
//   3: agents nearly standing still (slower than 1 cell per second)

#include "army_common.wgsl"

@group(0) @binding(6) var<storage, read_write> stats: array<atomic<u32>>;

const COUNTERS: u32 = 4u;

var<workgroup> local: array<atomic<u32>, 8>;

@compute @workgroup_size(8)
fn clear(@builtin(global_invocation_id) id: vec3u) {
	if (id.x < arrayLength(&stats)) {
		atomicStore(&stats[id.x], 0u);
	}
}

// Sums in workgroup memory first, so each workgroup adds to the global counters once
@compute @workgroup_size(256)
fn tally(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) lane: u32) {
	if (lane < TEAMS * COUNTERS) {
		atomicStore(&local[lane], 0u);
	}
	workgroupBarrier();

	if (id.x < arrayLength(&agents)) {
		let agent = agents[id.x];
		if (agent.state != 0u) {
			let base = agent.team * COUNTERS;
			let speed = length(agent.vel);
			atomicAdd(&local[base], select(0u, 1u, agent.goal == 1u));
			atomicAdd(&local[base + 1u], select(0u, 1u, agent.pos.x >= f32(GRID) * 0.5));
			atomicAdd(&local[base + 2u], u32(speed * 16.0));
			atomicAdd(&local[base + 3u], select(0u, 1u, speed < 1.0));
		}
	}
	workgroupBarrier();

	if (lane < TEAMS * COUNTERS) {
		atomicAdd(&stats[lane], atomicLoad(&local[lane]));
	}
}
