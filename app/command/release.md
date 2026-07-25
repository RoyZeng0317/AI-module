# RELEASE.md

## Release Checklist

### 程式

* [ ] Build Success
* [ ] Lint Passed
* [ ] Tests Passed

### 安全

* [ ] .env 未提交
* [ ] API Key 已檢查
* [ ] Debug Mode 已關閉

### 資料庫

* [ ] Migration 已測試
* [ ] 備份已完成

### 部署

* [ ] Docker Image 已建立
* [ ] Nginx Config 已更新
* [ ] SSL 正常

### 回滾

* [ ] 前一版 Image 保留
* [ ] Database Backup 可還原
* [ ] 回滾指令已驗證
