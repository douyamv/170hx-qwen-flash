#!/bin/bash
# 1) free the root disk (user-approved: delete the other models), 2) defragment the huihui shards on the slow disk by
# sequential re-copy, 3) build a root-disk model directory holding shards 1+4 (+MTP) with symlinks to shards 2/3
set -u
H=/mnt/slowdisk/AI-archive/models/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF; R=/home/douya/models-huihui
echo "REORG START $(date +%T)"; echo "root disk rotational: $(cat /sys/block/sdb/queue/rotational 2>/dev/null) (0 = SSD)"
rm -rf /home/douya/models /home/douya/models-fast; df -h / | tail -n 1
for n in 4 2 3; do
  f=$H/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000$n-of-00004.gguf; t0=$(date +%s)
  ionice -c3 nice -n 15 cp $f $f.contig && [ "$(stat -c %s $f)" = "$(stat -c %s $f.contig)" ] && mv -f $f.contig $f && echo "shard $n re-copied contiguously in $(( $(date +%s) - t0 )) s $(date +%T)" || { echo "shard $n COPY FAILED"; rm -f $f.contig; }
  if [ $n = 4 ]; then
    mkdir -p $R/UD-Q4_K_XL $R/MTP
    ionice -c3 cp $H/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf $R/UD-Q4_K_XL/
    ionice -c3 cp $f $R/UD-Q4_K_XL/ && echo "shard 4 copied to the root disk $(date +%T)"
    cp $H/MTP/mtp-Huihui-Qwen3.8-Flash-Next-shared-Q8_0-q4head.gguf $R/MTP/
    for m in 2 3; do ln -sf $H/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000$m-of-00004.gguf $R/UD-Q4_K_XL/; done
    ls -la $R/UD-Q4_K_XL/ | awk '{print $5, $9, $10, $11}'; df -h / | tail -n 1
  fi
done
sync; echo "REORG DONE $(date +%T)"; df -h / /mnt/slowdisk | tail -n 2
