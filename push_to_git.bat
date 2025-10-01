@echo off
cd /d "C:\Users\shubh\OneDrive\Desktop\blender pipeline"
git init
git remote add origin https://github.com/lekhaj/spark.git
git fetch origin
git checkout -b Shubham_v3_main
git add .
git commit -m "Add cleaned Blender pipeline"
git push -u origin Shubham_v3_main
pause

