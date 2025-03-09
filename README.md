# Recursive Reward Aggregation (RRA)

Code for Recursive Reward Aggregation.


## Videos for continuous control experiments



## Running Experiments



## Running with Docker
You can build and run the Docker container:
```sh
cd docker
docker build -t rra_image . -f Dockerfile 
```
### **Tips**
If you encounter issues with **Cython**, try the following:
```sh
pip uninstall Cython
pip install Cython==3.0.0a10
```
This can resolve version conflicts or compatibility issues with certain dependencies.
