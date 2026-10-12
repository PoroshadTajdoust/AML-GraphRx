# راهنمای سریع انتشار در GitHub

این بسته با نسخه‌های ارسالی مقاله و supplementary تطبیق داده شده است. فایل ZIP را ابتدا Extract کنید؛ محتوای پوشهٔ `AML-GraphRx` باید در ریشهٔ مخزن قرار بگیرد.

1. در GitHub از علامت `+` گزینهٔ **New repository** را انتخاب کنید و نام را `AML-GraphRx` بگذارید.
2. برای بررسی اولیه می‌توانید **Private** را انتخاب کنید؛ برای ارائهٔ لینک عمومی به ژورنال، مخزن باید در زمان انتشار **Public** باشد. انتشار خودکار انجام نشده است.
3. برای مسیر خط فرمان، مخزن را خالی بسازید؛ README یا license خودکار اضافه نکنید، چون README داخل بسته آماده است.
4. در پوشهٔ استخراج‌شده این دستورها را اجرا کنید. `YOUR_USERNAME` را با نام واقعی حساب خود جایگزین کنید:

```bash
git init
git add .
git commit -m "Release manuscript-linked analysis code and source tables"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/AML-GraphRx.git
git push -u origin main
```

اگر Git هویت نویسنده را درخواست کرد، نام و ایمیل خودتان را در تنظیمات آن وارد کنید. اطلاعات ورود یا token را داخل نوت‌بوک و فایل‌های مخزن قرار ندهید.

روش بدون خط فرمان: داخل مخزن **Add file → Upload files** را بزنید و فایل‌ها و پوشه‌های استخراج‌شده را بارگذاری کنید. GitHub در هر نوبت حداکثر ۱۰۰ فایل و برای هر فایل در مرورگر حداکثر ۲۵ MiB می‌پذیرد؛ این بسته را در چند نوبت بارگذاری کنید و ساختار پوشه‌ها را حفظ کنید. صرفاً بارگذاری ZIP باعث نمایش عادی کدهای مخزن نمی‌شود.

پیش از عمومی‌کردن، نام واقعی نویسندگان و مجوز نرم‌افزاری مورد تأیید آنان را اضافه کنید. DOI، آدرس مخزن و نام نویسنده در این بسته جعل نشده است. نام‌های داخلی فایل‌های نتایج برای اتصال صحیح به خروجی‌های اصلی حفظ شده‌اند؛ نام تحلیل‌ها در README علمی و توصیفی است.

کنترل سریع پس از استخراج:

```bash
python -m pip install -r requirements-replay.txt
python tests/check_release.py
python code/verify_key_results.py
```

`docs/SEED_CONCORDANCE.md` جدول کامل seedهاست. seed اصلی `20261004` و سه seed آموزش ablation برابر `20261031`، `20261101` و `20261102` است. این اعداد را به تاریخ روز یا seed مشترک ۴۲ تغییر ندهید.

بسته از ورودی‌های آماده و قفل‌شدهٔ پروژه شروع می‌شود. بازاجرای GPU و بازسازی کامل داده‌های خام در این مرحله انجام نشده است. فهرست بررسی‌های واقعاً انجام‌شده در `docs/VALIDATION.md` آمده است.

مستندات رسمی:
- https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository
- https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
