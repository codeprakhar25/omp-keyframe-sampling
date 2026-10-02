#!/usr/bin/env bash
# v8 C + inference frontend v0 (2026-10-02, step 2 after the C baseline box cbase). Same bad refs, lines, seeds and
# flags as fxen_c / fxpk_c; only the reference changes (scripts/aws_ref_frontend.py):
#   cfe = ref VAD-trimmed, internal pauses cut to <= 0.3 s, speech RMS -20 dBFS      (box fe)
#   cfd = same after spectral-gating denoise (noisereduce)                            (box fd)
# then scripts/aws_gap_cap.py writes a _gc copy of every arm (dead air capped after decode; the fe box also caps the
# raw C baseline arms) and the usual rulers score all of it.
#   logs/launch_cfront.sh plan fe|fd | up fe|fd
set -euo pipefail
REGION=us-east-1; BUCKET=real-voice-mio-692707725608; P=eval/incumbent_v1
SVHN="svh1 svh2 svh3 svh4 svh5 svh6 ctrl"; PKN="en1 en2 hi1 hi2 mx1 mx2"; SEEDS="7 11 23"
declare -A SVHF=([svh1]=svh_1 [svh2]=svh_2 [svh3]=svh_3 [svh4]=svh_4 [svh5]=svh_5 [svh6]=svh_6 [ctrl]=libritts_en_m_7127)
B="${2:-fe}"; [ "$B" = fe ] || [ "$B" = fd ] || { echo "box must be fe|fd"; exit 1; }
V=$([ "$B" = fe ] && echo fe || echo fedn); K=$([ "$B" = fe ] && echo cfe || echo cfd); TAG=cfront$B

join(){ local IFS=,; echo "$*"; }
en=(); enr=(); for n in $SVHN; do for s in $SEEDS; do en+=("fxen_${K}_${n}_s${s}"); enr+=("fxen_${K}_${n}_s${s}:en=${SVHF[$n]}_${V}:${s}"); done; done
pk=(); pkr=(); for n in $PKN; do for s in $SEEDS; do pk+=("fxpk_${K}_${n}_s${s}"); pkr+=("fxpk_${K}_${n}_s${s}:pk_${n}_${V}:${s}"); done; done
gc=("${en[@]}" "${pk[@]}")
if [ "$B" = fe ]; then for n in $SVHN; do for s in $SEEDS; do gc+=("fxen_c_${n}_s${s}"); done; done
  for n in $PKN; do for s in $SEEDS; do gc+=("fxpk_c_${n}_s${s}"); done; done; fi
gcd=(); for x in "${gc[@]}"; do gcd+=("${x}_gc"); done
EN=$(join "${en[@]}"); PK=$(join "${pk[@]}"); GC=$(join "${gc[@]}"); GCD=$(join "${gcd[@]}")
ENGC=$(join $(printf '%s\n' "${gcd[@]}" | grep '^fxen_')); PKGC=$(join $(printf '%s\n' "${gcd[@]}" | grep '^fxpk_'))
REFS="svh_1,svh_2,svh_3,svh_4,svh_5,svh_6,libritts_en_m_7127,pk_en1,pk_en2,pk_hi1,pk_hi2,pk_mx1,pk_mx2"
FLAGS="--ckpt runs/air_v8_c/ckpt_final --prefix $P --danda --chunk 0 --text-fixes --skip-done"

S="aws_ref_frontend.py --refs $REFS --variants $V"
S+=";aws_incumbent_ours.py $FLAGS --items items_fxen.json --langs en --runs $(join "${enr[@]}")"
S+=";aws_incumbent_ours.py $FLAGS --items items_fxpk.json --langs en,hi --runs $(join "${pkr[@]}")"
S+=";aws_gap_cap.py --arms $GC --prefix $P --skip-done"
S+=";aws_whisper_wer.py --arms $EN,$PK,$GCD --prefix $P --langs en"
S+=";aws_indicconformer_wer.py --arms $PK,$PKGC --prefix $P --name-tmpl inc_{arm} --langs hi"
S+=";aws_word_gaps.py --arms $EN,$PK,$GCD --prefix $P --langs en --skip-done"
S+=";aws_fx_sil.py --arms $EN,$PK,$GCD"
S+=";aws_fxen_sim.py --arms $EN,$ENGC --out fxensim_$K"
S+=";aws_fxen_sim.py --arms $PK,$PKGC --out fxpksim_$K"

up_now(){ local o; o=$(timeout 60 aws ec2 describe-instances --region $REGION --filters Name=tag:Name,Values=rv-ear-$TAG \
  Name=instance-state-name,Values=pending,running --query 'Reservations[].Instances[].InstanceId' --output text) || return 1; [ -n "$o" ]; }

launch_one(){ # launch_one type spot(1|"")
  local type=$1 spot=$2 ami az subnet iid extra=() ud
  ami=$(aws ssm get-parameter --region $REGION --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id --query Parameter.Value --output text)
  [ -n "$spot" ] && extra+=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}')
  ud=$(cat <<EOS
#!/bin/bash
export RV_BUCKET=${BUCKET} RV_REGION=${REGION} RV_MAX_HOURS=3 RV_TAG=${TAG} RV_OUT_PREFIX=${P} RV_TRAINSET=eval/neu_air_trainset_v8a2
mkdir -p /opt/rv/code
export RV_STEPS="\$(aws s3 cp s3://${BUCKET}/code/cfront_steps_${B}.txt - --region ${REGION})"
for f in aws_word_gaps.py aws_ref_frontend.py aws_gap_cap.py; do aws s3 cp s3://${BUCKET}/code/\$f /opt/rv/code/ --region ${REGION}; done
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
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=rv-ear-${TAG}},{Key=Project,Value=real-voice},{Key=Job,Value=v8c-frontend}]" \
      --instance-initiated-shutdown-behavior terminate --user-data "$ud" "${extra[@]}" \
      --query 'Instances[0].InstanceId' --output text 2>/dev/null) || iid=""
    [ -n "$iid" ] && [ "$iid" != None ] && { echo "instance $iid $type ${spot:+spot} $az"; return 0; }
    echo "  no $type ${spot:+spot} in $az"
  done
  return 1; }

case "${1:-plan}" in
plan) echo "$S" | tr ';' '\n' | cut -c1-240;;
up)
  # Scorers on s3 code/ already match the main checkout (md5 checked 2026-10-02); only the step list is new.
  echo "$S" | aws s3 cp - s3://$BUCKET/code/cfront_steps_${B}.txt --region $REGION --only-show-errors
  echo "start $(date -u +%T)"
  for i in $(seq 1 30); do
    up_now && { echo "UP $(date -u +%T)"; exit 0; }
    for tm in g5.xlarge:1 g6.xlarge:1 g5.xlarge: g6.xlarge:; do
      launch_one "${tm%%:*}" "${tm#*:}" && { sleep 20; up_now && { echo "UP $(date -u +%T) on $tm"; exit 0; }; }
    done
    sleep 240
  done
  echo "GAVE_UP $(date -u +%T)";;
*) echo "usage: $0 plan|up fe|fd"; exit 1;;
esac
