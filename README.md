# ima206_project

## Github Usage

1. Clone this [repository](https://github.com/Lupin2019/ima206_project.git)

2. create a new branch base on `main` branch

   ```shell
   git checkout main 
   # Output: Already on 'main'
   git pull 
   # Output: Already up to date.
   git checkout -b abc123 # (new branch name)
   # Output: Switched to a new branch 'abc123'
   
   # make some updates
   echo "Hello" > a.txt
   
   git add a.txt
   git commit -m "Add a new file ./a.txt"
   
   # Only the for first push:
   # add a new remote branch, typically keep the same name with the local one.
   git push --set-upstream origin abc123
   
   # Next push:
   git push 
   ```

