conda-forge

wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

bash Miniforge3-$(uname)-$(uname -m).sh



sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

cd to the dir

mamba env create -f conda_environment.yaml



batch 
{
obs: {image: tensor, agent_pos:tensor}
action: tensor

}