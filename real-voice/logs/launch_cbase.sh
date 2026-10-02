#!/usr/bin/env bash
# v8 C bad-ref baseline (user pick 2026-10-02, step 1 before the inference frontend).
# C = runs/air_v8_c/ckpt_final (v8 without IV-R). Its English Svarah bad-ref clips (fxen_c_*, 21 runs) were generated
# on 2026-09-28 but only svh1 was scored; it never ran on the user's laptop refs. This box:
#   1. generates fxpk_c_* with the exact fxpk_a recipe (items_fxpk.json, en+hi, --danda --chunk 0 --text-fixes, seeds 7/11/23)
#   2. scores fxen_c + fxpk_c with the same rulers as A / stock / CosyVoice / IndexTTS2: Whisper WER (en),
#      IndicConformer (hi), word gaps (en), fx_sil, WavLM SIM -> fxensim_c / fxpksim_c
#   logs/launch_cbase.sh plan | up
set -euo pipefail
REGION=us-east-1; BUCKET=real-voice-mio-692707725608; P=eval/incumbent_v1; TAG=cbase
SVHN="svh1 svh2 svh3 svh4 svh5 svh6 ctrl"; PKN="en1 en2 hi1 hi2 mx1 mx2"; SEEDS="7 11 23"

join(){ local IFS=,; echo "$*"; }
fxen=(); for n in $SVHN; do for s in $SEEDS; do fxen+=("fxen_c_${n}_s${s}"); done; done
fxpk=(); runs=(); for n in $PKN; do for s in $SEEDS; do fxpk+=("fxpk_c_${n}_s${s}"); runs+=("fxpk_c_${n}_s${s}:pk_${n}:${s}"); done; done
EN=$(join "${fxen[@]}"); PK=$(join "${fxpk[@]}"); ALL="$EN,$PK"

S="aws_incumbent_ours.py --ckpt runs/air_v8_c/ckpt_final --prefix $P --items items_fxpk.json --langs en,hi --danda --chunk 0 --text-fixes --skip-done --runs $(join "${runs[@]}")"
S+=";aws_whisper_wer.py --arms $ALL --prefix $P --langs en"
S+=";aws_indicconformer_wer.py --arms $PK --prefix $P --name-tmpl inc_{arm} --langs hi"
S+=";aws_word_gaps.py --arms $ALL --prefix $P --langs en"
S+=";aws_fx_sil.py --arms $ALL"
S+=";aws_fxen_sim.py --arms $EN --out fxensim_c"
S+=";aws_fxen_sim.py --arms $PK --out fxpksim_c"

up_now(){ local o; o=$(timeout 60 aws ec2 describe-instances --region $REGION --filters Name=tag:Name,Values=rv-ear-$TAG \
  Name=instance-state-name,Values=pending,running --query 'Reservations[].Instances[].InstanceId' --output text) || return 1; [ -n "$o" ]; }

launch_one(){ # launch_one type spot(1|"")
  local type=$1 spot=$2 ami az subnet iid extra=() ud
  ami=$(aws ssm get-parameter --region $REGION --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id --query Parameter.Value --output text)
  [ -n "$spot" ] && extra+=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}')
  ud=$(cat <<EOS
#!/bin/bash
export RV_BUCKET=${BUCKET} RV_REGION=${REGION} RV_MAX_HOURS=2 RV_TAG=${TAG} RV_OUT_PREFIX=${P} RV_TRAINSET=eval/neu_air_trainset_v8a2
mkdir -p /opt/rv/code
export RV_STEPS="\$(aws s3 cp s3://${BUCKET}/code/cbase_steps.txt - --region ${REGION})"
aws s3 cp s3://${BUCKET}/code/aws_word_gaps.py /opt/rv/code/ --region ${REGION}
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
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=rv-ear-${TAG}},{Key=Project,Value=real-voice},{Key=Job,Value=v8c-baseline}]" \
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
  echo "$S" | aws s3 cp - s3://$BUCKET/code/cbase_steps.txt --region $REGION --only-show-errors
  echo "start $(date -u +%T)"
  for i in $(seq 1 30); do
    up_now && { echo "UP $(date -u +%T)"; exit 0; }
    for tm in g5.xlarge:1 g6.xlarge:1 g5.xlarge: g6.xlarge:; do
      launch_one "${tm%%:*}" "${tm#*:}" && { sleep 20; up_now && { echo "UP $(date -u +%T) on $tm"; exit 0; }; }
    done
    sleep 240
  done
  echo "GAVE_UP $(date -u +%T)";;
*) echo "usage: $0 plan | up"; exit 1;;
esac
