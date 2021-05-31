#!/bin/bash

url=$1
payload=$2
content=${3:-application/json}

curl --data-binary @${payload} -H "Content-Type: ${content}" -v ${url}/invocations
