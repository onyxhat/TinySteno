"""macOS system audio capture via ScreenCaptureKit (macOS 12.3+).

Requires ``pyobjc-framework-ScreenCaptureKit`` and Screen Recording permission.
"""
# pylint: disable=no-name-in-module  # pyobjc framework members resolved at runtime on macOS
# pylint: disable=no-member  # pyobjc framework members resolved at runtime on macOS
# pylint: disable=invalid-name  # ObjC bridge method names use camelCase/underscore conventions
# pylint: disable=self-cls-assignment  # required ObjC init pattern: self = objc.super().init()
# pylint: disable=attribute-defined-outside-init  # ObjC init method is not __init__
# pylint: disable=access-member-before-definition  # ObjC init sets attrs; pylint can't see it
# pylint: disable=using-constant-test  # ctypes c_void_p truthiness check is intentional
# pylint: disable=global-statement  # module-level cache pattern for lazy-loaded C library
# pylint: disable=too-many-locals  # complex platform-specific hardware interface functions
# pylint: disable=too-many-branches  # complex platform-specific hardware interface functions
# pylint: disable=consider-using-with  # Semaphore.acquire() is not a resource-allocating op

import ctypes
import ctypes.util
import threading
from typing import Callable, Optional

import numpy as np


# ── CoreMedia ctypes wrappers ─────────────────────────────────────────────────

class _AudioBuffer(ctypes.Structure):
    _fields_ = [
        ("mNumberChannels", ctypes.c_uint32),
        ("mDataByteSize", ctypes.c_uint32),
        ("mData", ctypes.c_void_p),
    ]


def _load_coremedia() -> ctypes.CDLL:
    path = ctypes.util.find_library("CoreMedia")
    if not path:
        raise OSError("CoreMedia library not found")
    lib = ctypes.CDLL(path)

    lib.CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer.restype = ctypes.c_int32
    lib.CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer.argtypes = [
        ctypes.c_void_p,                   # sbuf
        ctypes.POINTER(ctypes.c_size_t),   # bufferListSizeNeededOut (nullable)
        ctypes.c_void_p,                   # bufferListOut (nullable for size query)
        ctypes.c_size_t,                   # bufferListSize
        ctypes.c_void_p,                   # blockBufferStructureAllocator
        ctypes.c_void_p,                   # blockBufferBlockAllocator
        ctypes.c_uint32,                   # flags
        ctypes.POINTER(ctypes.c_void_p),   # blockBufferOut
    ]
    lib.CFRelease.restype = None
    lib.CFRelease.argtypes = [ctypes.c_void_p]

    lib.CMSampleBufferGetFormatDescription.restype = ctypes.c_void_p
    lib.CMSampleBufferGetFormatDescription.argtypes = [ctypes.c_void_p]
    lib.CMAudioFormatDescriptionGetStreamBasicDescription.restype = ctypes.c_void_p
    lib.CMAudioFormatDescriptionGetStreamBasicDescription.argtypes = [ctypes.c_void_p]

    return lib


_CM: Optional[ctypes.CDLL] = None


def _detect_sr(sb_ptr_val: int) -> Optional[float]:
    """Read the actual sample rate from a CMSampleBuffer's format description."""
    global _CM
    if _CM is None:
        _CM = _load_coremedia()
    try:
        fmt = _CM.CMSampleBufferGetFormatDescription(ctypes.c_void_p(sb_ptr_val))
        if not fmt:
            return None
        # CMAudioFormatDescriptionGetStreamBasicDescription returns a pointer to
        # AudioStreamBasicDescription whose first field is Float64 mSampleRate.
        asbd = _CM.CMAudioFormatDescriptionGetStreamBasicDescription(ctypes.c_void_p(fmt))
        if not asbd:
            return None
        return ctypes.c_double.from_address(asbd).value
    except Exception:
        return None


def _sb_to_float32(sample_buffer) -> Optional[np.ndarray]:
    """Extract interleaved float32 samples from a CMSampleBuffer (pyobjc object)."""
    global _CM
    if _CM is None:
        _CM = _load_coremedia()

    # CMSampleBufferRef arrives as a PyObjCPointer; the raw address is in
    # .pointerAsInteger (a read-only C member exposed by pyobjc-core).
    if hasattr(sample_buffer, "pointerAsInteger"):
        sb_ptr = ctypes.c_void_p(sample_buffer.pointerAsInteger)
    else:
        try:
            sb_ptr = sample_buffer.__c_void_p__()
        except AttributeError:
            return None
    if not sb_ptr:
        return None

    # Step 1: query the size needed for the AudioBufferList
    needed = ctypes.c_size_t(0)
    _CM.CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sb_ptr, ctypes.byref(needed), None, 0, None, None, 0, None
    )
    if needed.value == 0:
        return None

    # Step 2: allocate and fill the AudioBufferList
    abl_raw = ctypes.create_string_buffer(needed.value)
    block_buf = ctypes.c_void_p(0)
    err = _CM.CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sb_ptr, None, abl_raw, needed.value, None, None, 0,
        ctypes.byref(block_buf),
    )
    if err != 0:
        return None

    # Step 3: parse AudioBufferList
    # Layout: UInt32 mNumberBuffers | <alignment padding> | AudioBuffer[n]
    ab_align = ctypes.alignment(_AudioBuffer)
    header_size = (4 + ab_align - 1) & ~(ab_align - 1)  # round up to alignment
    ab_size = ctypes.sizeof(_AudioBuffer)

    n_buffers = ctypes.c_uint32.from_buffer_copy(abl_raw.raw[:4]).value
    channels: list[np.ndarray] = []
    for i in range(n_buffers):
        off = header_size + i * ab_size
        ab = _AudioBuffer.from_buffer_copy(abl_raw.raw[off : off + ab_size])
        if ab.mData and ab.mDataByteSize > 0:
            n_ch = max(ab.mNumberChannels, 1)
            n_frames = ab.mDataByteSize // (4 * n_ch)
            raw = np.ctypeslib.as_array(
                (ctypes.c_float * (n_frames * n_ch)).from_address(ab.mData)
            ).copy()
            if n_ch == 1:
                channels.append(raw)
            else:
                # interleaved: split into per-channel arrays
                interleaved = raw.reshape(n_frames, n_ch)
                for c in range(n_ch):
                    channels.append(interleaved[:, c])

    if block_buf.value:
        _CM.CFRelease(block_buf)

    if not channels:
        return None

    n = min(len(c) for c in channels)
    # mix all channels down to mono for loopback use
    mono = np.mean(np.column_stack([c[:n] for c in channels]), axis=1)
    return mono.reshape(-1, 1)


# ── SCStreamOutput delegate ───────────────────────────────────────────────────

_DelegateClass = None  # created lazily so pyobjc import is optional


def _get_delegate_class():
    global _DelegateClass
    if _DelegateClass is not None:
        return _DelegateClass

    import objc
    import ScreenCaptureKit as SCK
    from Foundation import NSObject

    class _AudioDelegate(NSObject, protocols=[objc.protocolNamed("SCStreamOutput")]):
        def initWithCallback_onRate_(self, cb, rate_cb):
            self = objc.super(_AudioDelegate, self).init()
            if self is None:
                return None
            self._cb = cb
            self._rate_cb = rate_cb
            self._rate_reported = False
            return self

        @objc.typedSelector(b"v@:@^{opaqueCMSampleBuffer=}q")
        def stream_didOutputSampleBuffer_ofType_(  # pylint: disable=unused-argument  # ObjC protocol requires this signature
            self, _stream, sample_buffer, output_type
        ):
            if output_type != SCK.SCStreamOutputTypeAudio:
                return
            try:
                if not self._rate_reported and self._rate_cb is not None:
                    if hasattr(sample_buffer, "pointerAsInteger"):
                        sr = _detect_sr(sample_buffer.pointerAsInteger)
                        if sr:
                            self._rate_cb(sr)
                            self._rate_reported = True
                arr = _sb_to_float32(sample_buffer)
                if arr is not None and self._cb is not None:
                    self._cb(arr)
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug("SCStreamOutput callback error: %s", e)

        def stream_didStopWithError_(self, stream, error):
            pass

    _DelegateClass = _AudioDelegate
    return _DelegateClass


# ── Public API ────────────────────────────────────────────────────────────────

class MacOSLoopback:
    """Capture macOS system audio via ScreenCaptureKit.

    Requires macOS 12.3+, Screen Recording permission, and
    ``pyobjc-framework-ScreenCaptureKit`` to be installed.
    """

    def __init__(self, sample_rate: int, callback: Callable[[np.ndarray], None]):
        self.sample_rate = sample_rate
        self.callback = callback
        self.detected_sample_rate: Optional[float] = None
        self._stream = None
        self._ready = threading.Event()
        self._error: Optional[Exception] = None

    def start(self) -> None:
        """Start system audio capture. Blocks until the stream is running."""
        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()
        if not self._ready.wait(timeout=8.0):
            exc = self._error or RuntimeError("Timed out starting ScreenCaptureKit stream")
            raise exc
        if self._error:
            raise self._error

    def stop(self) -> None:
        """Stop system audio capture."""
        stream, self._stream = self._stream, None
        if stream is None:
            return
        done = threading.Event()
        stream.stopCaptureWithCompletionHandler_(lambda *_: done.set())
        done.wait(timeout=3.0)

    # ── private ──────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:  # pylint: disable=too-many-statements  # complex platform-specific ScreenCaptureKit setup sequence
        try:
            import time
            import ScreenCaptureKit as SCK
            from Foundation import NSRunLoop, NSDate
            import warnings
            import objc as _objc
            # CMSampleBufferRef is intentionally handled as a PyObjCPointer
            warnings.filterwarnings("ignore", category=_objc.ObjCPointerWarning)

            # ── Get the main display via SCShareableContent ────────────────────
            # We spin the run loop while waiting so the XPC completion callback
            # is delivered on this thread (avoids the pyobjc dispatch issue).
            content_ready = threading.Event()
            content_holder = [None]

            def on_content(*args):
                for arg in args:
                    if arg is not None and hasattr(arg, "displays"):
                        content_holder[0] = arg
                        break
                content_ready.set()

            SCK.SCShareableContent.getShareableContentWithCompletionHandler_(on_content)

            run_loop = NSRunLoop.currentRunLoop()
            deadline = time.monotonic() + 5.0
            while not content_ready.is_set() and time.monotonic() < deadline:
                run_loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))

            if not content_ready.is_set() or content_holder[0] is None:
                raise RuntimeError("Timed out waiting for SCShareableContent")
            displays = content_holder[0].displays()
            if not displays:
                raise RuntimeError("No displays returned by SCShareableContent")
            display = displays[0]

            # Exclude nothing → capture all apps/windows on this display
            init_filter = (
                SCK.SCContentFilter.alloc()
                .initWithDisplay_excludingApplications_exceptingWindows_
            )
            filt = init_filter(display, [], [])
            if filt is None:
                raise RuntimeError("SCContentFilter init failed")

            cfg = SCK.SCStreamConfiguration.alloc().init()
            cfg.setCapturesAudio_(True)
            cfg.setExcludesCurrentProcessAudio_(False)
            cfg.setSampleRate_(self.sample_rate)
            cfg.setChannelCount_(1)
            # Minimise video overhead — we only want audio
            cfg.setWidth_(2)
            cfg.setHeight_(2)

            DelegateClass = _get_delegate_class()
            delegate = DelegateClass.alloc().initWithCallback_onRate_(
                self.callback,
                lambda sr: setattr(self, "detected_sample_rate", sr),
            )

            self._stream = SCK.SCStream.alloc().initWithFilter_configuration_delegate_(
                filt, cfg, None
            )
            self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
                delegate, SCK.SCStreamOutputTypeAudio, None, None
            )

            # ── Start the stream ───────────────────────────────────────────────
            # pyobjc dispatches startCaptureWithCompletionHandler_ callbacks
            # with 0 Python args (full_signature b'@@@' hides both block ptr
            # and NSError under the method-stub convention), so use *args.
            sem = threading.Semaphore(0)

            def on_start(*_args):
                # _args is empty — no reliable error info from pyobjc here;
                # failure surfaces as stream producing no audio samples.
                self._ready.set()
                sem.release()

            self._stream.startCaptureWithCompletionHandler_(on_start)
            sem.acquire(timeout=8.0)

            if not self._ready.is_set():
                raise RuntimeError("SCStream startCapture timed out")

            # Keep spinning the run loop so SCK callbacks keep firing
            run_loop = NSRunLoop.currentRunLoop()
            while self._stream is not None:
                run_loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))

        except Exception as exc:
            self._error = exc
            self._ready.set()
