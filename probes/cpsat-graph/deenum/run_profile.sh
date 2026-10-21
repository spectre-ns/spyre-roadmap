#!/bin/bash
# Stage-0 profile: one cold process per zoo model, only while the card is free.
D="$(cd "$(dirname "$0")" && pwd)"
TS="${TORCH_SPYRE:-$HOME/torch-spyre}"
cd "$TS"
export PYTHONPATH="$TS:$PYTHONPATH"
R=$D/runs
mkdir -p $R/errs
MODELS="softmax_256x1024 softmax_1024x2048 rmsnorm_128x256 rmsnorm_512x2048 residual_rmsnorm_256x1024 row_stats_512x1024 pointwise_chain_512x1024 softplus_128x64 linear_bias_gelu_256x1024x4096 mm_chain_256x512x1024x256 mlp_gelu_256x1024x4096 mlp_swiglu_128x512x1536 mlp_swiglu_256x1024x4096 mlp_stack_128x512x2048 simple_attn_4x64x256x128 sdpa_1x8x256x64 sdpa_1x4x512x128 sdpa_causal_1x8x128x128 sdpa_1x12x512x128 sdpa_1x32x512x64 sdpa_1x16x1024x64 sdpa_1x8x2048x64 sdpa_4x8x256x64 sdpa_causal_1x12x512x128 sdpa_causal_1x8x1024x64 sdpa_gqa_1x32x8x512x128 sdpa_gqa_1x16x4x1024x64 sdpa_q64_kv1024_1x8x64 attn_block_128x512x8 attn_block_256x1024x16 transformer_block_128x512x8x1536 transformer_block_256x1024x16x2816"

held() { for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q "/dev/vfio" && return 0; done; return 1; }
FREE_SINCE=0
replay() {  # STOP|CONT the CPU-heavy replay, if one is running
  local pid; pid=$(cat $R/REPLAY_PID 2>/dev/null)
  [ -n "$pid" ] && kill -$1 $pid 2>/dev/null
}
acquire() {  # card free on two checks 1 s apart, replay stopped
  while :; do
    if held; then replay CONT; FREE_SINCE=0; sleep 2; continue; fi
    [ $FREE_SINCE = 0 ] && FREE_SINCE=$(date +%s)
    date +%s%N > $R/GEN
    replay STOP
    sleep 1
    held || return 0
  done
}
unit() {  # cfg key cmd...
  local cfg=$1 key=$2; shift 2
  grep -qF "$cfg $key	ok" $R/profile_done.txt 2>/dev/null && return 0
  acquire
  if [ $cfg = tiling ]; then export AUTO_COARSE_TILING=1; else unset AUTO_COARSE_TILING; fi
  local out
  out=$(timeout 900 "$@" 2>$R/errs/$cfg.$key.err | tail -1)
  echo "$(date +%T) $cfg $key	$out" >> $R/profile_log.txt
  case "$out" in *"	ok	"*) echo "$cfg $key	ok" >> $R/profile_done.txt;; esac
}
for pass in 1 2; do
  for cfg in default tiling; do
    for m in $MODELS; do
      unit $cfg $m python3 $D/zoo_t.py $m $R/profile_$cfg.jsonl $R/caps_$cfg
    done
  done
done
# Granite needs ~3 min of card; only start it after the card sat idle 60 s.
for cfg in default tiling; do
  grep -qF "$cfg granite	ok" $R/profile_done.txt 2>/dev/null && continue
  for try in $(seq 1 240); do
    acquire
    [ $(( $(date +%s) - FREE_SINCE )) -ge 60 ] && break
    sleep 10
  done
  replay CONT
  [ $(( $(date +%s) - FREE_SINCE )) -ge 60 ] || { echo "$(date +%T) $cfg granite skipped: card never idle 60 s" >> $R/profile_log.txt; continue; }
  unit $cfg granite python3 $D/granite_t.py ibm-granite/granite-4.0-micro granite_micro 2 $R/profile_granite_$cfg.jsonl $R/caps_granite_$cfg
done
replay CONT
echo PROFILE_DONE
