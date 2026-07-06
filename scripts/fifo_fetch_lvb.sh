#!/usr/bin/env bash
# Fast LongVideoBench subset fetch: aria2c downloads each ~5.2GB part with 16 parallel
# connections (defeats HF's per-connection throttle -> full bandwidth) into a single
# persistent `tar` via a FIFO. Disk holds ~1 part + the extracted subset (fits 100GB).
# tar exits early once the last wanted member is found -> we stop downloading remaining parts.
set -uo pipefail
cd /workspace/slm-lab
set -a; . ./.env; set +a
OUT=data/videos
MEMBERS=$OUT/_members.txt
BASE=https://huggingface.co/datasets/longvideobench/LongVideoBench/resolve/main
mkdir -p "$OUT"

# regenerate members file from the manifest (videos/<file> lines)
python3 -c "
import json
d=json.load(open('data/manifest.lvb.json'))
open('$MEMBERS','w').write(''.join(f'videos/{m}\n' for m in sorted({x['video_file'] for x in d})))
"
echo "members: $(wc -l < "$MEMBERS")"

FIFO=/tmp/tarfifo
rm -f "$FIFO"; mkfifo "$FIFO"

tar -x -C "$OUT" --strip-components=1 --occurrence -T "$MEMBERS" < "$FIFO" 2>"$OUT/_tar.err" &
TARPID=$!
# persistent writer fd -> tar never sees EOF until we close fd 3 at the very end
# (without this, each `cat > FIFO` closes the pipe and tar exits after ONE part)
exec 3>"$FIFO"

parts=""
for a in a b c; do for b in a b c d e f g h i j k l m n o p q r s t u v w x y z; do parts="$parts $a$b"; done; done
parts=$(echo $parts | tr ' ' '\n' | head -31)

i=0
for suf in $parts; do
  i=$((i+1))
  kill -0 $TARPID 2>/dev/null || { echo "tar exited (all members found) after $((i-1)) parts"; break; }
  echo "[$i/31] downloading part.$suf ..."
  aria2c -x16 -s16 -j16 -k1M --allow-overwrite=true --file-allocation=none -q \
    --header="Authorization: Bearer $HF_TOKEN" \
    -d /tmp -o part.bin "$BASE/videos.tar.part.$suf"
  if [ $? -ne 0 ]; then echo "aria2 failed on $suf"; break; fi
  cat /tmp/part.bin >&3 2>/dev/null || { echo "tar closed pipe (done)"; rm -f /tmp/part.bin; break; }
  rm -f /tmp/part.bin
  echo "   fed part.$suf; videos so far: $(ls "$OUT"/*.mp4 2>/dev/null | wc -l)"
done

# close persistent writer so tar sees EOF
exec 3>&-
wait $TARPID 2>/dev/null
echo "FIFO_FETCH_DONE videos=$(ls "$OUT"/*.mp4 2>/dev/null | wc -l)"
