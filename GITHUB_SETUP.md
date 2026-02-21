# Как выложить проект на GitHub

## 1. Создай репозиторий на GitHub

1. Зайди на https://github.com/new  
2. **Repository name** — например `ML-in-Math` или `pancake-ml`  
3. **Public**, без галочек «Add README» / «Add .gitignore»  
4. Нажми **Create repository**

## 2. В папке проекта выполни в терминале

Подставь вместо `ТВОЙ_USERNAME` и `ИМЯ_РЕПО` свои данные с GitHub.

```bash
cd "C:\Users\Иван Литвак\Desktop\ML in Math"

git init
git add .
git commit -m "Initial commit: Pancake problem solvers, baseline, Kaggle pipeline"

git remote add origin https://github.com/ТВОЙ_USERNAME/ИМЯ_РЕПО.git
git branch -M main
git push -u origin main
```

При первом `git push` GitHub может запросить логин: используй **Personal Access Token** вместо пароля (Settings → Developer settings → Personal access tokens).

После успешного пуша этот файл можно удалить или оставить для справки.
