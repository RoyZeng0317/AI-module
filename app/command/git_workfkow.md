# GIT_WORKFLOW.md

## 開發流程

1. git pull
2. 建立 feature branch
3. 修改程式
4. git add
5. git commit
6. git push
7. 建立 PR

## 禁止指令

* git push --force
* git reset --hard origin/main
* git clean -fdx
* git rebase -i main（未經確認）

## Commit 規範

* feat:
* fix:
* refactor:
* docs:
* test:
* chore:

## 每次提交前

請執行：

git status
git diff --staged
