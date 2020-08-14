docker run --gpus all --name tennis --rm -it -v "$(pwd)":/rl -p 8900:8888 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 aalawami/banana_rl:cuda10.2 --
