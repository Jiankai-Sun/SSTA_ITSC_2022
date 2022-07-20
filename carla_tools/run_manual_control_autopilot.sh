# bash $HOME/data/Apps/CARLA_0.9.11/CarlaUE4.sh &
CURR_DIR=$(pwd)
cd $HOME/data/Apps/CARLA_0.9.11/
bash CarlaUE4.sh &
sleep 5
cd $CURR_DIR
echo $CURR_DIR
# sleep 5
# conda activate carla
# python no_rendering_mode.py --show-spawn-points # --show-triggers
#python manual_control.py --autopilot --recording True --img_save_dir _out_1 &
#sleep 15
#python manual_control.py --autopilot --recording True --img_save_dir _out_2 &
#sleep 15
#python manual_control.py --autopilot --recording True --img_save_dir _out_3 &
#sleep 15
#python manual_control.py --autopilot --recording True --img_save_dir _out_4 &
python spawn_npc.py -n 100 -w 0 --sync
#python spawn_npc_path_follow.py -n 1 -w 0 --sync -s 0
#python manual_control.py
#python automatic_control.py