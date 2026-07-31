# Shared worker-pool helper, sourced by the stage runners.
#
# We deliberately avoid `xargs -P N -I{}`: on macOS (BSD xargs) a constructed
# argument may not exceed 255 bytes, and our job commands are longer than that,
# so xargs silently refuses the whole batch with "command line cannot be
# assembled, too long". A plain bash job-control pool has no such limit.
run_pool() {
  local max="$1" jobsfile="$2" cmd
  while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$max" ]; do sleep 3; done
    bash -c "$cmd" &
  done < "$jobsfile"
  wait
}
