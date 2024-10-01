conda-forge

wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

bash Miniforge3-$(uname)-$(uname -m).sh

`pip install --upgrade pyqt5_tools`


sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

cd to the dir

mamba env create -f conda_environment.yaml

pip install einops-exts

batch 
{
obs: {image: tensor, agent_pos:tensor}
action: tensor

}

(obs, action)???

agent_pose torch.Size([64, 16, 2])
tensor([378.8397, 393.6135, 405.3241, 413.1841, 417.6390, 419.9036, 420.7098,
        420.9152, 419.7893, 417.0476, 412.9117, 408.1582, 403.2375, 398.5620,
        392.1198, 385.3951], device='cuda:0')
agent_action torch.Size([64, 16, 2])
tensor([414., 420., 422., 422., 422., 421., 421., 417., 412., 406., 401., 396.,
        392., 381., 376., 373.], device='cuda:0')


how to see the architecture of vision encoder

what is the condition_mask

why there is no