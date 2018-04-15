#!/usr/bin/env bash

basepath=$(cd `dirname $0`; pwd)
#echo "you select" $1
cd $basepath
if [ $1 = "yolo_valid" ] ; then
    ./darknet detector valid train_cfg/detector.data train_cfg/detector_valid.cfg model/yolo-weight.weights results/

elif [ $1 = "yolo_train" ] ; then
./darknet detector train train_cfg/detector.data train_cfg/detector.cfg model/darknet19_448.conv.23

elif [ $1 = "classifier_train" ] ; then
./darknet classifier train train_cfg/classifier.data train_cfg/classifier.cfg

elif [ $1 = "classifier_valid" ] ; then
./darknet classifier valid train_cfg/classifier.data train_cfg/classifier_valid.cfg model/chinese_character.weights
