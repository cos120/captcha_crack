#!/usr/bin/env bash

basepath=$(cd `dirname $0`; pwd)
#echo "you select" $1
cd $basepath
if [ $1 = "yolo_valid" ] ; then
./darknet detector valid cfg/yolo-valid.data cfg/yolo-valid.cfg model/yolo-weight.weights results/

elif [ $1 = "yolo_train" ] ; then
./darknet detector train cfg/voc-origin.data cfg/yolo-origin.cfg model/darknet19_448.conv.23

elif [ $1 = "classifier_train" ] ; then
./darknet classifier train cfg/chinese.data cfg/chinese_character.cfg

elif [ $1 = "classifier_valid" ] ; then
./darknet classifier valid cfg/chinese.data cfg/chinese_character.cfg model/chinese_character.weights
