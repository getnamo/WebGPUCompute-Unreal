// The army example (Content/Examples/Army) driven from JavaScript instead of Blueprint.
//
// The same WGSL files run the simulation; this script owns the GPU resources, dispatches the kernels each
// frame, and reads per-team numbers back from the GPU (army_stats.wgsl) to show them. Both hot reload:
//   - save a .wgsl file: the kernels recompile in place, the army keeps going
//   - save this file: the script runs again; its scope releases everything the old run made first
//
// Setup: BP_WebGPUArmyJs (map ArmyExampleJs) runs this as the DefaultScript of a JavascriptInstanceComponent
// with the features UnrealClasses, UnrealMemory, UnrealGlobals, UnrealMisc, Context and JsOwner. The owning
// actor provides ArmyFloor, ObstacleCapture, TitleText, StatsText, DisplayTarget and ReloadCount, and sends
// 'key' 'R' through the instance component's Emit. (Give Blueprint members distinctive names: scripts see
// them spelled as the engine's name table first saw the word, so a member called Floor can show up as floor.)

// ---------------- settings: edit and save ----------------

const AGENTS = 131072;      // agents in both armies (army_common.wgsl's Agent is 32 bytes)
const RELAX_STEPS = 24;     // flow field relaxation passes per frame
const TITLE = 'WebGPU army, driven by JavaScript (examples/webgpu/army.js)';

// ---------------------------------------------------------

const gpu = require('webgpu');

const SHADERS = '/WebGPUCompute/Examples/Army';
const GRID = 256;           // army_common.wgsl's GRID and DISPLAY
const DISPLAY = 1024;
const TEAMS = 2;
const TEAM_NAMES = ['orange', 'blue'];

const owner = JsOwner;

// 131072 -> "131,072" (this V8 has no Intl, so toLocaleString doesn't group digits)
const grouped = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
const scope = gpu.scope('army.js');

// Reload or end of play: free everything this run made
Context.OnEndPlay = () => scope.release();

// The floor shows the display render target
owner.ArmyFloor.CreateDynamicMaterialInstance(0, null, 'None').SetTextureParameterValue('SlateUI', owner.DisplayTarget);

// Resources, by the names the shaders use
const bindings = scope.bindings('army');
bindings.buffer('agents', AGENTS, 32);
bindings.buffer('field', TEAMS * GRID * GRID, 4);
bindings.buffer('density', TEAMS * GRID * GRID, 4);
const stats = bindings.buffer('stats', TEAMS * 4, 4);
const display = scope.texture(owner.DisplayTarget, 'army display');
const obstacles = scope.texture(owner.ObstacleCapture.TextureTarget, 'army obstacles');
bindings.bind('display', display).bind('obstacles', obstacles);

const kernel = (file, entry) => scope.kernel({ file: `${SHADERS}/${file}`, entry, bindings });
const prepare = kernel('army_field.wgsl', 'prepare');
const count = kernel('army_field.wgsl', 'count');
const relax = kernel('army_field.wgsl', 'relax');
const simulate = kernel('army_behavior.wgsl', 'simulate');
const respawn = kernel('army_behavior.wgsl', 'respawn');
const clearStats = kernel('army_stats.wgsl', 'clear');
const tally = kernel('army_stats.wgsl', 'tally');

// The numbers come back a frame or two later, read in place from the mapped readback
const statsStream = scope.stream(stats, { slots: 3, policy: 'Latest' });

owner.TitleText.K2_SetText(TITLE);

let time = 0;
let frameSeconds = 1 / 60;
let sinceText = 1;
let latest = null;

Context.OnTick = (deltaSeconds) => {
	time += deltaSeconds;
	frameSeconds += (deltaSeconds - frameSeconds) * 0.05;
	bindings.set('params.dt', deltaSeconds).set('params.time', time);

	gpu.access([display, obstacles], () => {
		prepare.dispatch(DISPLAY, DISPLAY);
		count.dispatch(AGENTS);
		relax.dispatch(GRID, GRID, TEAMS, RELAX_STEPS);
		simulate.dispatch(AGENTS);
	});
	clearStats.dispatch(TEAMS * 4);
	tally.dispatch(AGENTS);
	statsStream.capture();

	statsStream.latest((bytes, info) => {
		const counters = new Uint32Array(bytes);
		latest = {
			teams: TEAM_NAMES.map((name, team) => {
				const base = team * 4;
				const size = AGENTS / TEAMS;
				return {
					name,
					marchingEast: counters[base],
					eastHalf: counters[base + 1],
					speed: counters[base + 2] / 16 / size,
					still: counters[base + 3],
				};
			}),
			framesLate: info.FramesLate,
			roundTripMs: info.RoundTripMs,
		};
	});

	sinceText += deltaSeconds;
	if (latest && sinceText >= 0.25) {
		sinceText = 0;
		const lines = latest.teams.map((team) =>
			`${team.name}: ${grouped(team.marchingEast)} marching east, ${grouped(team.eastHalf)} in the east half, ` +
			`${team.speed.toFixed(1)} cells/s, ${grouped(team.still)} standing`);
		lines.push(`${grouped(AGENTS)} agents   ${(1 / frameSeconds).toFixed(0)} fps   stats ${latest.framesLate} frames behind ` +
			`(${latest.roundTripMs.toFixed(2)} ms GPU round trip)   script reload #${owner.ReloadCount}`);
		owner.StatsText.K2_SetText(lines.join('<br>'));
	}
};

// [R]: everyone back to the start (the Blueprint forwards the key)
Context.OnMessage = (name, message) => {
	if (name === 'key' && message === 'R') {
		respawn.dispatch(AGENTS);
	}
};

console.log(`army.js: ${AGENTS} agents, kernels from ${SHADERS}`);
