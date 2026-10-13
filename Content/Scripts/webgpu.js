// WebGPU compute from JavaScript (UnrealJs), on the WebGPUCompute plugin.
//
//   const gpu = require('webgpu');
//   const scope = gpu.scope('demo');               // owns what it makes: scope.release() frees all of it
//   const bindings = scope.bindings();             // resources every kernel made with it gets by WGSL name
//   const values = bindings.buffer('values', 1024, 4);
//   const kernel = scope.kernel({ source: WGSL, entry: 'main', bindings });   // or { file: '/Game/Shaders/x.wgsl' }
//   await kernel.ready;                            // the first successful compile
//   gpu.write(values, new Float32Array(1024).fill(1));
//   bindings.set('params.scale', 2);               // a member of `var<uniform> params`
//   kernel.dispatch(1024);
//   const result = new Float32Array(await gpu.read(values));
//
// Every frame, a readback stream brings results back without copying them:
//   const stream = scope.stream(values);           // 3 slots, Latest policy
//   stream.capture();                              // after this frame's dispatches
//   stream.latest((bytes, info) => {               // an earlier frame, viewed in place in mapped GPU memory;
//       const v = new Float32Array(bytes);         // the view is detached when the callback returns
//   });
//
// Kernels from .wgsl files hot reload as in Blueprint. Release the scope when the script reloads or ends:
//   Context.OnEndPlay = () => scope.release();                        // JavascriptInstanceComponent
//   module.exports = () => { ...; return () => scope.release(); };   // devrequire

'use strict';

const USAGE = { storage: 1, uniform: 2, indirect: 4 };

const hasMemory = () => typeof memory !== 'undefined' && typeof memory.access === 'function';

/** The UObject behind a wrapper (Bindings, Kernel, Stream) or the object itself. */
const native = (object) => (object && object.native) ? object.native : object;

function available() {
	return typeof WebGPUJs !== 'undefined' && WebGPUJs.IsAvailable();
}

/** Writes a typed array, DataView or ArrayBuffer into a buffer at byteOffset (multiples of 4). */
function write(buffer, data, byteOffset = 0) {
	const view = ArrayBuffer.isView(data) ? data
		: data instanceof ArrayBuffer ? new Uint8Array(data)
		: new Float32Array(data);
	if (hasMemory() && available()) {
		// Straight from script memory into the upload of the next submit
		let ok = false;
		memory.exec(view.buffer, () => {
			ok = WebGPUJs.WriteBufferFromScript(native(buffer), byteOffset, view.byteOffset, view.byteLength);
		});
		return ok;
	}
	native(buffer).WriteBytes(new Uint8Array(view.buffer, view.byteOffset, view.byteLength), byteOffset);
	return true;
}

/** Reads byteLength bytes (all by default) after every earlier command: a Promise of an ArrayBuffer (a copy). */
function read(buffer, byteOffset = 0, byteLength = -1) {
	return new Promise((resolve, reject) => {
		const request = available() ? WebGPUJs.ReadBuffer(native(buffer), byteOffset, byteLength) : null;
		if (!request) {
			reject(new Error('webgpu.read: no buffer, or WebGPU is unavailable'));
			return;
		}
		request.OnCompleted.Add((ok) => {
			if (!ok) {
				request.Release();
				reject(new Error('webgpu.read: the read failed (device lost?)'));
				return;
			}
			const lent = request.GetMemory();
			const bytes = lent && hasMemory() ? memory.access(lent).slice(0) : Uint8Array.from(request.CopyBytes()).buffer;
			request.Release();
			resolve(bytes);
		});
	});
}

/** Runs fn between BeginAccess and EndAccess of shared textures (render targets shared with the renderer). */
function access(textures, fn) {
	const list = (Array.isArray(textures) ? textures : [textures]).map(native).filter(Boolean);
	list.forEach((texture) => texture.BeginAccess());
	try {
		return fn();
	} finally {
		list.forEach((texture) => texture.EndAccess());
	}
}

class Bindings {
	constructor(object) {
		this.native = object;
	}

	/** A new zeroed buffer of count * stride bytes, bound as name. */
	buffer(name, count, stride = 4, usage = USAGE.storage) {
		return this.native.CreateBuffer(name, count, stride, usage);
	}

	/** An existing buffer, or a texture, bound as name. */
	bind(name, resource) {
		const object = native(resource);
		if (object && typeof object.BeginAccess === 'function') {
			this.native.SetTexture(name, object);
		} else {
			this.native.SetBuffer(name, object);
		}
		return this;
	}

	/** A uniform value by path ("params.dt"): a number, or an array / {x, y, z, w} for vectors. */
	set(name, value) {
		if (typeof value === 'number') {
			this.native.SetFloat(name, value);
		} else if (Array.isArray(value) || ArrayBuffer.isView(value)) {
			this.native.SetVector(name, { X: value[0] || 0, Y: value[1] || 0, Z: value[2] || 0, W: value[3] || 0 });
		} else if (value && typeof value === 'object') {
			this.native.SetVector(name, { X: value.x || value.X || 0, Y: value.y || value.Y || 0, Z: value.z || value.Z || 0, W: value.w || value.W || 0 });
		} else {
			throw new Error(`webgpu: can't set ${name} to ${value}`);
		}
		return this;
	}

	/** An integer uniform value, exact beyond 2^24. */
	setInt(name, value) {
		this.native.SetInt(name, value | 0);
		return this;
	}

	get(name) {
		return this.native.GetBuffer(name);
	}
}

class Kernel {
	constructor(object, label) {
		this.native = object;
		this.label = label;
		this.listeners = [];
		let resolveReady;
		/** Resolves with this kernel at its first successful compile (later edits that fail don't affect it). */
		this.ready = new Promise((resolve) => { resolveReady = resolve; });
		this.native.OnCompiled.Add((ok, messages) => {
			if (ok) {
				resolveReady(this);
			} else {
				console.error(`webgpu: ${label} doesn't compile:\n${messages}`);
			}
			this.listeners.forEach((listener) => listener(ok, messages));
		});
		if (this.native.IsReady()) {
			resolveReady(this);
		}
	}

	/** fn(ok, messages) after every compile, including hot reloads. */
	onCompiled(fn) {
		this.listeners.push(fn);
		return this;
	}

	/** Enough workgroups for x * y * z invocations, run repeat times in a row. */
	dispatch(x, y = 1, z = 1, repeat = 1) {
		this.native.DispatchForCount(x, y, z, repeat);
		return this;
	}

	/** Exact workgroup counts. */
	dispatchGroups(x, y = 1, z = 1, repeat = 1) {
		this.native.Dispatch(x, y, z, repeat);
		return this;
	}

	/** Binds a buffer or texture to this kernel only, by WGSL name. */
	bind(name, resource) {
		const object = native(resource);
		if (object && typeof object.BeginAccess === 'function') {
			this.native.SetTextureByName(name, object);
		} else {
			this.native.SetBufferByName(name, object);
		}
		return this;
	}

	get isReady() {
		return this.native.IsReady();
	}

	/** Errors and warnings of the latest compile. */
	get messages() {
		return this.native.GetCompileMessages();
	}
}

class Stream {
	constructor(object, source) {
		this.native = object;
		this.source = source;
	}

	/** Copies the source (or byteLength bytes of a buffer from byteOffset) into a free slot, after the commands before it. False when dropped. */
	capture(byteOffset = 0, byteLength = -1) {
		const source = native(this.source);
		return typeof source.BeginAccess === 'function' ? this.native.CaptureTexture(source) : this.native.Capture(source, byteOffset, byteLength);
	}

	/**
	 * reader(bytes, info) with the next frame (by the stream's policy) viewed in place: no copy. The view is
	 * detached when reader returns; copy what you keep. Returns reader's result, or undefined when no frame is ready.
	 */
	latest(reader) {
		if (!hasMemory() || !available()) {
			const result = this.native.ReadBytes();
			return result.$ ? reader(Uint8Array.from(result.OutBytes).buffer, this.native.GetLastFrameInfo()) : undefined;
		}
		const frame = WebGPUJs.AcquireFrame(this.native);
		if (!frame) {
			return undefined;
		}
		try {
			return reader(memory.access(frame), this.native.GetLastFrameInfo());
		} finally {
			frame.Release();
		}
	}

	/** The next frame to hold across calls: { bytes, info, release() }. Its slot stays in use until release(). Null when none is ready. */
	acquire() {
		const frame = available() && hasMemory() ? WebGPUJs.AcquireFrame(this.native) : null;
		return frame ? { bytes: memory.access(frame), info: this.native.GetLastFrameInfo(), release: () => frame.Release() } : null;
	}

	get readyCount() {
		return this.native.GetReadyCount();
	}

	stats() {
		return this.native.GetStats();
	}
}

/** Owns GPU objects so a script can free them together (on reload, EndPlay, or whenever it's done). */
class Scope {
	constructor(name) {
		this.name = name || 'script';
		this.owned = [];
	}

	track(object, release) {
		this.owned.push({ object, release });
		return object;
	}

	bindings(label) {
		const object = WebGPUBlueprintLibrary.CreateWebGPUBindings(label || this.name);
		if (!object) {
			throw new Error('webgpu: WebGPU is unavailable');
		}
		return this.track(new Bindings(object), (bindings) => bindings.native.Release());
	}

	/** A buffer of byteLength bytes (usage: gpu.usage flags). */
	buffer(byteLength, usage = USAGE.storage, label = '') {
		const object = WebGPUBlueprintLibrary.CreateWebGPUBuffer(byteLength, usage, label || this.name);
		if (!object) {
			throw new Error('webgpu: WebGPU is unavailable');
		}
		return this.track(object, (buffer) => buffer.Release());
	}

	/**
	 * { source | file, entry = 'main', bindings, hotReload = true, label }. Files: content-style paths
	 * ('/Game/Shaders/x.wgsl', '/PluginName/...'), absolute, or relative to the project; they hot reload.
	 */
	kernel(options) {
		const entry = options.entry || 'main';
		const bindings = native(options.bindings) || null;
		const label = options.label || (options.file ? `${options.file.split('/').pop()} (${entry})` : `${this.name} (${entry})`);
		let object;
		if (options.file) {
			object = WebGPUBlueprintLibrary.CreateWebGPUKernelFromFile(options.file, entry, bindings, options.hotReload !== false, label);
		} else {
			object = WebGPUBlueprintLibrary.CreateWebGPUKernel(options.source, entry, label);
			if (object && bindings) {
				object.SetBindings(bindings);
			}
		}
		if (!object) {
			throw new Error('webgpu: WebGPU is unavailable');
		}
		return this.track(new Kernel(object, label), (kernel) => kernel.native.Release());
	}

	/** A render target shared with kernels without copies (D3D12): set Shared, and Supports UAV to write it. */
	texture(renderTarget, label = '') {
		const object = WebGPUBlueprintLibrary.CreateWebGPUTextureFromRenderTarget(renderTarget, label);
		if (!object) {
			throw new Error('webgpu: the render target can\'t be shared (Shared enabled? D3D12?)');
		}
		return this.track(object, (texture) => texture.Release());
	}

	/** A readback stream over a buffer or texture: { slots = 3, policy = 'Latest' | 'Queue' | 'Lockstep' }. */
	stream(source, options = {}) {
		const slots = options.slots || 3;
		const policy = options.policy || 'Latest';
		const object = native(source);
		const created = typeof object.BeginAccess === 'function'
			? WebGPUBlueprintLibrary.CreateWebGPUReadbackStreamForTexture(object, slots, policy, options.label || '')
			: WebGPUBlueprintLibrary.CreateWebGPUReadbackStreamForBuffer(object, slots, policy, options.label || '');
		if (!created) {
			throw new Error('webgpu: no readback stream for that source');
		}
		return this.track(new Stream(created, object), (stream) => stream.native.Release());
	}

	/** Frees everything this scope made, newest first. Safe to call more than once. */
	release() {
		const owned = this.owned.reverse();
		this.owned = [];
		for (const { object, release } of owned) {
			try {
				release(object);
			} catch (e) {
				console.error(`webgpu: releasing ${this.name}: ${e}`);
			}
		}
	}
}

function scope(name) {
	return new Scope(name);
}

module.exports = { scope, write, read, access, available, usage: USAGE, Scope, Bindings, Kernel, Stream };
