docker run --gpus all --name reacher --rm -it -v "$(pwd)":/rl -p 8900:8888 aalawami/banana_rl:cuda10.2 --
