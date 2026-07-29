# MEMORIZE.md

# AI Memory Policy

Version: 1.0

---

# Purpose

本文件定義 AI 在專案中的長期記憶策略。

AI 應區分：

- 長期知識（Long-Term Memory）
- 工作記憶（Working Memory）
- 暫存資訊（Temporary Context）

避免將短期內容永久保留。

---

# Memory Priority

依照以下優先順序記憶。

Priority 1
Project Rules

Priority 2
Architecture

Priority 3
Coding Style

Priority 4
Business Logic

Priority 5
Developer Preference

Priority 6
Task Progress

Priority 7
Temporary Conversation

---

# Long-Term Memory

可以長期記住：

## Project

- Project Name
- Project Purpose
- Main Features
- Technology Stack
- Folder Structure

---

## Coding Style

- Naming Convention
- Folder Convention
- API Convention
- State Management
- Error Handling
- Logging Style

---

## Developer Preference

例如：

- 使用繁體中文回答
- 程式碼註解使用英文
- Commit 使用 Conventional Commit
- 優先考慮效能
- 避免過度抽象化

---

## Architecture

記住：

- Domain
- Application
- Infrastructure
- Presentation

以及：

- Dependency Rule
- Clean Architecture
- SOLID

---

## Business Rules

例如：

股票不得使用未驗證資料

金流不得略過驗證

登入流程不得修改

API Version 不可變更

---

# Working Memory

工作記憶只保留：

目前 Issue

目前 PR

目前 Branch

目前 Sprint

目前 Bug

目前 Feature

完成後應清除。

---

# Temporary Memory

以下內容不得永久保存：

Debug Log

Terminal Output

Build Error

Compiler Error

Stack Trace

Screenshot

測試資料

一次性 Prompt

---

# Forget Policy

以下情況立即遺忘：

Issue 已完成

PR 已 Merge

Branch 已刪除

Task 已完成

Debug 已結束

Session 結束

---

# Memory Update Rule

當以下內容變更時：

Architecture

Folder

Database

API

Coding Style

Developer Preference

請更新記憶。

---

# Memory Conflict

若新資訊與舊資訊衝突：

1.
確認來源

2.
詢問使用者

3.
更新記憶

不得自行猜測。

---

# Memory Safety

不得記住：

Password

API Key

Token

Cookie

Credential

Private Key

Secret

OTP

信用卡資訊

身份證資訊

個人敏感資料

---

# Memory Compression

若資訊重複：

請合併。

例如：

不要保留：

Flutter 使用 Riverpod

Flutter 使用 Riverpod 2

Flutter 使用 Riverpod 最新版

而改成：

Flutter 使用 Riverpod。

---

# Memory Validation

更新記憶前請確認：

是否仍有效

是否仍被使用

是否已有更新版本

是否與 Architecture 衝突

---

# Memory Confidence

每項記憶請附帶：

High

Medium

Low

若信心不足：

請詢問使用者。

---

# Preferred Recall Order

AI 回憶資訊時依照：

Architecture

↓

Business Rule

↓

Coding Style

↓

Developer Preference

↓

Current Task

↓

Conversation

---

# Memory Format

請使用：

Category

Item

Reason

Confidence

Last Updated

Example

---

# Review Policy

每次大型修改後：

請重新整理：

Project Structure

API

Database

Dependency

Architecture

Coding Style

並更新必要記憶。

---

# Final Rule

AI 應：

優先更新重要知識。

不要永久保存暫時資訊。

不要猜測。

不知道就詢問。

保持記憶簡潔、一致、可維護。