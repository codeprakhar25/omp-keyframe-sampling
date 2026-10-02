#!/usr/bin/env bash
# DNSMOS census of the v8 training sources (user pick 2026-10-02, data policy: DNSMOS on every source + per-source
# table). One spot xlarge: decode ~400 clips per source x lang from eval/neu_air_trainset_v8a2 (v8 C sources + IV-R),
# score DNSMOS P.835 + brouhaha SNR/C50 -> s3 eval/data_audit_v1/dnsmos_census_v8/{clips.jsonl,summary.json}.
#   logs/launch_dnsmos.sh plan | up
set -euo pipefail
REGION=us-east-1; BUCKET=real-voice-mio-692707725608; P=eval/data_audit_v1; TAG=dnsmos
S="aws_dnsmos_census.py --trainset eval/neu_air_trainset_v8a2 --per-group 400"

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
export RV_STEPS="\$(aws s3 cp s3://${BUCKET}/code/dnsmos_steps.txt - --region ${REGION})"
aws s3 cp s3://${BUCKET}/code/aws_dnsmos_census.py /opt/rv/code/ --region ${REGION}
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
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=rv-ear-${TAG}},{Key=Project,Value=real-voice},{Key=Job,Value=dnsmos-census}]" \
      --instance-initiated-shutdown-behavior terminate --user-data "$ud" "${extra[@]}" \
      --query 'Instances[0].InstanceId' --output text 2>/dev/null) || iid=""
    [ -n "$iid" ] && [ "$iid" != None ] && { echo "instance $iid $type ${spot:+spot} $az"; return 0; }
    echo "  no $type ${spot:+spot} in $az"
  done
  return 1; }

case "${1:-plan}" in
plan) echo "$S" | tr ';' '\n' | cut -c1-240;;
up)
  aws s3 cp "$(dirname "$0")/../scripts/aws_dnsmos_census.py" s3://$BUCKET/code/ --region $REGION --only-show-errors
  echo "$S" | aws s3 cp - s3://$BUCKET/code/dnsmos_steps.txt --region $REGION --only-show-errors
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
