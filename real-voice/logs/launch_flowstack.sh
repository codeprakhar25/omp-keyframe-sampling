#!/usr/bin/env bash
# Flow-stack probe (user pick 2026-09-29): stock CosyVoice3 (cz = ref in LM, cx = ref only in flow) and IndexTTS2
# (ix = pooled ref) on the same Svarah bad refs (fxen, 6 lines) and user laptop-mic refs (fxpk, en lines) x seeds
# 7/11/23 as the stock-Air (fxen_sh / fxpk_sh) and v8 A (fxen_a / fxpk_a) arms. Two spot xlarge boxes, one per system.
#   logs/launch_flowstack.sh plan        print both step lists
#   logs/launch_flowstack.sh up cv|ix    launch one box (loops over types / AZs, stops once it is up)
# Run from the flowstack-probe worktree; existing scorers are uploaded from the main real-voice checkout.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"          # worktree real-voice/
MAIN=/home/prakh/ml-resarch/real-voice             # main checkout: bootstrap + existing scorers
REGION=us-east-1; BUCKET=real-voice-mio-692707725608; P=eval/incumbent_v1
SVH=svh_1,svh_2,svh_3,svh_4,svh_5,svh_6,libritts_en_m_7127; SVHN=svh1,svh2,svh3,svh4,svh5,svh6,ctrl
PK=pk_en1,pk_en2,pk_hi1,pk_hi2,pk_mx1,pk_mx2;              PKN=en1,en2,hi1,hi2,mx1,mx2

arms(){ # arms <out-prefix> <modes> <names>
  local o="$1" m n s out=(); for m in ${2//,/ }; do for n in ${3//,/ }; do for s in 7 11 23; do out+=("${o}_${m}_${n}_s${s}"); done; done; done
  local IFS=,; echo "${out[*]}"; }
baseline(){ local o=() a n s; for a in sh a; do for n in ${SVHN//,/ }; do for s in 7 11 23; do o+=("fxen_${a}_${n}_s${s}"); done; done
  for n in ${PKN//,/ }; do for s in 7 11 23; do o+=("fxpk_${a}_${n}_s${s}"); done; done; done; local IFS=,; echo "${o[*]}"; }

steps(){ # steps cv|ix
  local sys modes; [ "$1" = cv ] && { sys=cosyvoice3; modes=cz,cx; } || { sys=indextts2; modes=ix; }
  local A; A="$(arms fxen "$modes" "$SVHN"),$(arms fxpk "$modes" "$PKN")"
  local S="aws_flowstack_probe.py --system $sys --modes $modes --items items_fxen.json --refs $SVH --names $SVHN --out-prefix fxen"
  S+=";aws_flowstack_probe.py --system $sys --modes $modes --items items_fxpk.json --langs en --refs $PK --names $PKN --out-prefix fxpk"
  S+=";aws_whisper_wer.py --arms $A --prefix $P --langs en"
  S+=";aws_word_gaps.py --arms $A --prefix $P --langs en"
  S+=";aws_fx_sil.py --arms $A"
  S+=";aws_fxen_sim.py --arms $A --out fxensim_$1"
  # Same gap metric on the existing stock-Air + v8 A arms, so every arm is read with one ruler (ix box: fewer clips).
  [ "$1" = ix ] && S+=";aws_word_gaps.py --arms $(baseline) --prefix $P --langs en --skip-done"
  echo "$S"; }

up_now(){ local o; o=$(timeout 60 aws ec2 describe-instances --region $REGION --filters Name=tag:Name,Values=rv-ear-flow$1 \
  Name=instance-state-name,Values=pending,running --query 'Reservations[].Instances[].InstanceId' --output text) || return 1; [ -n "$o" ]; }

launch_one(){ # launch_one cv|ix type spot(1|"")
  local tag=flow$1 type=$2 spot=$3 ami az subnet iid extra=()
  ami=$(aws ssm get-parameter --region $REGION --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id --query Parameter.Value --output text)
  [ -n "$spot" ] && extra+=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}')
  local ud; ud=$(cat <<EOS
#!/bin/bash
export RV_BUCKET=${BUCKET} RV_REGION=${REGION} RV_MAX_HOURS=3 RV_TAG=${tag} RV_OUT_PREFIX=${P} RV_TRAINSET=eval/neu_air_trainset_v8a2
mkdir -p /opt/rv/code
export RV_STEPS="\$(aws s3 cp s3://${BUCKET}/code/flowstack_steps_$1.txt - --region ${REGION})"
for f in aws_flowstack_probe.py aws_word_gaps.py; do aws s3 cp s3://${BUCKET}/code/\$f /opt/rv/code/ --region ${REGION}; done
aws s3 cp s3://${BUCKET}/code/aws_air_infer_bootstrap.sh /root/boot.sh --region ${REGION}
bash /root/boot.sh
EOS
)
  for az in $(aws ec2 describe-instance-type-offerings --region $REGION --location-type availability-zone --filters Name=instance-type,Values=$type --query 'InstanceTypeOfferings[].Location' --output text | tr '\t' '\n' | sort -u); do
    subnet=$(aws ec2 describe-subnets --region $REGION --filters Name=default-for-az,Values=true Name=availability-zone,Values=$az --query 'Subnets[0].SubnetId' --output text)
    [ "$subnet" = None ] && continue
    iid=$(aws ec2 run-instances --region $REGION --image-id "$ami" --instance-type $type --count 1 --subnet-id "$subnet" \
      --iam-instance-profile Name=RealVoiceTrainerRole --metadata-options HttpTokens=required,HttpEndpoint=enabled \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=rv-ear-${tag}},{Key=Project,Value=real-voice},{Key=Job,Value=flowstack-probe}]" \
      --instance-initiated-shutdown-behavior terminate --user-data "$ud" "${extra[@]}" \
      --query 'Instances[0].InstanceId' --output text 2>/dev/null) || iid=""
    [ -n "$iid" ] && [ "$iid" != None ] && { echo "instance $iid $type ${spot:+spot} $az tag $tag"; return 0; }
    echo "  no $type ${spot:+spot} in $az"
  done
  return 1; }

case "${1:-plan}" in
plan) for b in cv ix; do echo "== $b"; steps $b | tr ';' '\n' | cut -c1-220; done;;
up)
  b="${2:?cv|ix}"
  for f in "$HERE/scripts/aws_flowstack_probe.py" "$HERE/scripts/aws_word_gaps.py"; do
    aws s3 cp "$f" s3://$BUCKET/code/ --region $REGION --only-show-errors; done
  for f in aws_air_infer_bootstrap.sh aws_whisper_wer.py aws_fx_sil.py aws_v8_sil_audit.py aws_fxen_sim.py; do
    aws s3 cp "$MAIN/scripts/$f" s3://$BUCKET/code/ --region $REGION --only-show-errors; done
  # Step list goes via S3: ~100 arm names x 4 scorers would crowd the 16 KB user-data limit.
  steps "$b" | aws s3 cp - s3://$BUCKET/code/flowstack_steps_$b.txt --region $REGION --only-show-errors
  echo "start $(date -u +%T)"
  for i in $(seq 1 30); do
    up_now "$b" && { echo "UP $(date -u +%T)"; exit 0; }
    for tm in g5.xlarge:1 g6.xlarge:1 g5.xlarge: g6.xlarge:; do
      launch_one "$b" "${tm%%:*}" "${tm#*:}" && { sleep 20; up_now "$b" && { echo "UP $(date -u +%T) on $tm"; exit 0; }; }
    done
    sleep 240
  done
  echo "GAVE_UP $(date -u +%T)";;
*) echo "usage: $0 plan | up cv|ix"; exit 1;;
esac
