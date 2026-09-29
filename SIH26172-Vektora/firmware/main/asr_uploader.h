// asr_uploader.h — record after a wake and POST the PCM to the ASR server.
//
// ==========================================================================
// THE ONE DESIGN DECISION THAT MATTERS: this module never touches I2S.
// ==========================================================================
//
// capture_task in main.cc owns I2S_NUM_0 exclusively. It sits in
// i2s_channel_read(..., portMAX_DELAY) in a tight loop and there is no idle
// window on that channel. A second reader does not "share" the stream, it
// SPLITS it: each reader gets an arbitrary subset of the samples. That would
// corrupt the recording and the keyword spotter's feature ring at the same
// time, and it would look like a model regression rather than a plumbing bug.
//
// So instead of reading the microphone, this module TAPS the samples that
// capture_task has already read, via asr_feed(). One line in the capture
// loop, append-only, cannot alter the KWS data path.
//
// Three things fall out of that, all of them wins:
//
//   1. The keyword spotter never pauses. The feature ring stays continuous,
//      so there is no gap to flush and no bogus inference on resume.
//   2. The audio sent to the server is bit-identical to what the KWS heard --
//      same >> CONFIG_VK_SAMPLE_SHIFT conversion, same clipping.
//   3. We get the PRE-ROLL that CONTEXT.md requires ("it also sends the audio
//      it recorded BEFORE it decided"). Detection is not instantaneous: at a
//      measured 564 ms per inference plus a ~500 ms decision period, the
//      verdict can land ~1.1 s after the keyword ended, by which point the
//      speaker is already into the command. A module that starts recording at
//      the hook would clip it. The tap keeps a rolling second, so we send it.
//
// Memory: the PCM is STREAMED to the server as it arrives, not buffered whole.
// A 4 s buffer would be 128 kB of contiguous internal RAM, which this board
// does not have spare once the tensor arena has taken its 128 kB block. The
// ring below is 48 kB and the HTTP body is written out of it in 4 kB chunks.

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocate the ring and start the uploader task. Call once at boot, AFTER
// model_start() (see the heap-ordering note in asr_uploader.c) and BEFORE
// capture_task is created, so the ring exists before the first asr_feed().
//
// On failure it logs and returns the error; asr_feed() and
// asr_capture_and_send() then become harmless no-ops, so a failed init
// cannot take the keyword spotter down with it.
esp_err_t asr_init(void);

// Feed the samples capture_task just read from I2S. Called once per 10 ms
// frame from the capture loop -- the single line added to main.cc.
//
// Cost: one memcpy of 320 bytes into a circular buffer, ~2 us. Against the
// measured 4.10 ms/frame the feature extraction already spends, that is noise.
// Never blocks, never allocates, safe before asr_init() (no-op).
void asr_feed(const int16_t *samples, int n);

// THE HOOK. Call from the wake handler.
//
// Returns in microseconds -- it only marks the byte range to send and wakes
// the uploader task. The recording and the POST happen on that task, on its
// own 8 kB stack. Deliberately asynchronous:
//
//   * the wake handler lives inside infer_task, whose 8 kB stack is sized for
//     the model, not for lwip + esp_http_client (which want ~6 kB between
//     them). Running the upload there risks a stack overflow.
//   * "never blocks forever" is then guaranteed at the call site by
//     construction rather than by getting every timeout right.
//
// A call made while an upload is already running is ignored (and logged).
// WiFi is OFF while listening: the uploader task turns the radio on after the
// wake, streams from the ring (which kept recording during the connect), then
// turns it off again. A failed connect degrades to "LED works, no transcript";
// the wake handler itself never waits on the network.
//
// The arguments go into the UDP HELLO so the server can compute L1/L2 and
// find the keyword in the stream (docs/protocol.md).
void asr_capture_and_send(int64_t t_win_end_us, int64_t t_detect_us, float score);

// Audio sent so far in the current (or last) wake, in ms. For the OLED.
uint32_t asr_sent_ms(void);

// True from the moment asr_capture_and_send() is accepted until the upload
// has finished or failed. This is the flag you asked for. Note that with the
// tap design the KWS does NOT need to pause -- see asr_uploader.c. It is
// exposed so you can suppress detections during an upload if you want to.
bool asr_is_busy(void);

typedef enum { ASR_IDLE = 0, ASR_LINKING, ASR_STREAMING } asr_state_t;
asr_state_t asr_state(void);

// Uploads aborted because the mic lapped the uploader (must be 0 in demos).
uint32_t asr_overruns(void);

// The ring doubles as the audio history for the gate's feature backfill
// (CLAUDE.md §5.2: one ring, three consumers). Sample index = samples fed
// since boot. Copies n samples starting at first_sample; false if any of
// them are not written yet or already overwritten. Safe from any task.
bool asr_ring_read(uint64_t first_sample, int16_t *dst, int n);

#ifdef __cplusplus
}
#endif
