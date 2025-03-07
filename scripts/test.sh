
jump_num_data=0
if [ -n "$jump_num_data" ]; then
    jump_num_data_arg="--jump_num_data $jump_num_data"
fi

echo "$jump_num_data_arg"