# HPC repo setup — cloning capstone-scripts

Clone this repo by using the following. (If it doesn't work follow the tutorial below
```sh
cd /fred/oz479/COS4005-P37-Capstone-Project
git clone ssh://git@github.com/shrimp332/capstone-scripts <your username>
```


> [!WARNING]
> **Use SSH, not HTTPS.** Every git operation on this repo must go over SSH.
> The clone URL is `ssh://git@github.com/shrimp332/capstone-scripts` — if you find
> yourself typing or pasting anything starting with `https://github.com/`, stop.

## Requirements
So because we're using SSH instead of HTTPS (see above warning) we have to do the following, how to is below.
1. Generate SSH keys on the HPC
2. Upload your keys to github (In the account settings - not the repo settings)

## Tutorial

### 1. Set user.name & user.email
```sh
git config --global user.name <your name>
git config --global user.email <email of your choice>
```
Note github does provide you with a pseudo email, mine for example is: 101687942+shrimp332@users.noreply.github.com  
I don't remember how to find this though.

### 2. Generate SSH Key
run the following on the HPC
```sh
ssh-keygen
```
This will generate keys for you, all of the defaults are fine so you should be able to just hit **enter** until your shell prompt appears again
### 3. Copy your new key to github
```sh
cat ~/.ssh/id_*.pub
```
this should print out your public key to your terminal. Copy it and go to https://github.com/settings/keys.  
1. Click **New SSH key**  
![New SSH key button](docs/Pasted%20image%2020260825164415.png)

2. Enter your new key into the text box  

![Pasting the key into the text box](docs/Pasted%20image%2020260825164514.png)

3. Set key type to Authentication.  

You should now be able to run authenticated git commands on the HPC.

## New Changes to the actual workspace
We're using git now and we're using seperate workspaces so that we don't work over each other. Large files that will not be put into git will remain in the Capstone folder but the scripts will be duplicated into your own seperate workspaces.  
Here is a pseudo example of the workspaces:
```
├── Fully_trained/
├── user1/
│   ├── Fully_trained -> /home/lmcd/test2/Fully_trained/
│   └── script.py
├── user2/
│   ├── Fully_trained -> /home/lmcd/test2/Fully_trained/
│   └── script.py
└── user3/
    ├── Fully_trained -> /home/lmcd/test2/Fully_trained/
    └── script.py

```

> [!IMPORTANT]
> when you finish making a change make a commit and push
> remember to git pull often OR make a seperate branch and open a pull request Luke will merge them very fast just @ me on discord in the GC
