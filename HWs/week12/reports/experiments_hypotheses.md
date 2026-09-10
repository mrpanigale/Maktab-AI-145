
## Part 1:

1.1- انتخاب Loss مناسب مسئله:<br> بر اساس جدول موجود در فایل نوت بوک cross_entropy برای مسائلی که بیش از 2 کلاس دارند و هر کلاس دقیقا به یک شی اشاره دارد و چند لیبل ندارد انتخاب درست Cross Entropy است برای همین در این مسئله که این شروط برای کلاس های آن صدق میکند از این معیار استفاده خواهم کرد . <br>
1.2- نتایج baseline model: <br>
Manual Loss: 1.7846660614013672 <br>
PyTorch loss: 1.7846659421920776 <br>
1/10 log(worst loss for 10 class): 2.3025850929940455 <br>
<br>
اگر 9 کلاس داشتیم نتایج به این صورت میشد چون در حالت اول هر کلاس در بدترین حالت 10 درصد احتمال داشت اینجا هر کلاس در بدترین حالت حداکثر 11 درصد شانس دارد تقریبا برای همین اعداد متفاوت شدند <br>
Manual Loss: 1.5360091924667358<br>
1/10 log(worst loss for 10 class): 2.198225077669803<br>
 PyTorch loss: 1.5360091924667358<br>

## Part 2:
1.1 : پیش بینی شکل data در طی مسیر شبکه CNN <br>
```shape
input
[B, 1, 28, 28]
conv 1
[B, 16, 28, 28]
relu
[B, 16, 28, 28]
maxpool
[B, 16, 14, 14]
conv 2
[B, 32, 14, 14]
relu
[B, 32, 14, 14]
maxpool
[B, 32, 7, 7]
Flatten
[B, 32 × 7 × 7 = 1568]
Linear
[B, 10]

```
1.2 : شکل درست برای داده های ما 32×7×7 است ولی همکار به اشتباهی 8×8 در نظر گرفته که احتمالا مربوط به شیپ ورودی  CIFAR باشد<br>
```Error
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x1568 and 2048x10)
```
اگر شیپ ورودی لایه آخر توسط همکار ما اشتباه انتخاب شود در همان ابتدا که داده ها به شبکه داده شود ، با این خطا مواجه میشود و حتی شبکه به محاسبه لاس و دادن خروجی هم نمیرسد
## Part3: 
روی 10 عدد از تصاویر آگمنتیشن زدم و پلات مربوطه را در ریپورت ها ذخیره کردم ، در ادامه بر این اساس به سوالات مربوط به آگمنتیشن پاسخ میدهم <br>

| استراتژی | حفظ لیبل                     | واقع‌گرایی                        | حفظ اطلاعات                     | تصمیم نهایی                                                       |
| :--- |:-----------------------------|:----------------------------------|:--------------------------------|:------------------------------------------------------------------|
| استراتژی ۱ | بله                          | بله                               | بله                             | مناسب                                                             |
| استراتژی ۲ | کمی گمراه کننده              | تغییر زاویه 90 درجه در محصول غلطه | بله                             | مناسب چون ممکن است شلوار روی زمین پهن شده و تصویر برداری شده باشد |
| استراتژی ۳ | بله شکل هندسی به هم نمی خورد | بله                               | جزئیات برخی تصاویر از بین رفته  | با تنظیم مناسب بله ، چون برخی تصاویر بیش از حد تاریک شده بودند    |


## Regularized(dropout 20%):
![loss_curves.png](plots/loss_curves.png)
تنها بر اساس  نمودار لاس قصد تحلیل ندارم با مراجعه به گزارش های موجود در reports میتوان دید که هیچ یک از مدل ها اورفیت نشده اند ولی مدل بیس لاین اندکی گپ بیشتری در اکیوریسی ترین و تست دارد <br>
حالا با توجه به نمودار لاس برای سه آزمایش مدل آگمنتد لاس بیشتری داشته ولی اختلاف کمتری بین ترین و تست آن وجود داشته همچنین مدل رگولارایزد در جایگاه دوم قرار دارد و طبق نمودار هم بیشترین اختلاف لاس برای مدل بیس بوده که هیچ آگمنتیشن و رگولاریزیشنی روی آن انجام نشده .

## Part 4:
4.1 : شکل ورودی Linear در  Head برابر تعداد کانال خروجی آخرین بخش کانولوشنی هست (32 )چرا که AdaptiveAverage طول و عرض آن را به 1 تبدیل کرد و تنها بچ سایز و تعداد کانال باقی ماند.<br>
4.3 :<br>
result Plain model first epoch--> train loss 0.80 , val loss 0.55<br>
result residual model first epoch--> train loss 1.57 , val loss 1.20<br>
خیر با شهود ما تطبیق نداشت ، چرا که انتظار میرفت مدل رزنت با شروع بهتری پیش برود چرا که بهتر جریان گرادیان را انتقال میدهد ولی این طور نبود و حتی عملکرد کلی مدل ساده از مدل رزنت بهتر بود؛ حتی از تمام مدل ها بهتر بود، البته به شدت دچار بیش برازش شده بود<br>
4.4 : زمانی که به علت افزایش پیچیدگی و عمق مدل انتقال گرادیان با مشکل مواجه میشود و اپتیمایزر نمی تواند به خوبی پارامتر ها را آپدیت کند دقت ترین و تست همزمان کاهش پیدا میکند و به کاهش دقت همزمان روی ترین و تست ناشی از افزایش عمق Degradation problem  میگویند <br>
## Part 5: 
```code
#هارد کد کردن تعداد کلاس اشتباه هست باید len(class_names) میداد
# در ضمن مدل رو به Device نبرده 
final_model = TinyResNetFashion(num_classes=10) # مشکل۱

# برای داده های ترین دیتا لودر ساخته نشده 
train_aug_full = datasets.FashionMNIST(
root="data", train=True, download=True, transform=train_augmentation
)
#داده های تست یه زیر مجموعه رندوم از دادههای ترین هست و روش آگمنتیشن هم خورده که کاملا اشتباهه
leaky_test_loader = DataLoader(
Subset(train_aug_full, range(TEST_LIMIT)), batch_size=64, shuffle=True
# مشکل۲و۳(دومشکل
جدا در همین خط)
)
optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
# ترین لودر همون طور که بالاتر اشاره شد اصلا ساخته نشده 
for images, labels in train_loader:
# عکس ها رو به دیوایس برده ولی لیبل ها رو نبرده Runtime Error
    images = images.to(DEVICE) # مشکل۴
    # نباید به کراس انتروپی لاس سافت مکس میداد باید لاجیت خام میداد چون به صورت درونی اعمال میشه سافت مکس داخل خود تابع 
    probs = torch.softmax(final_model(images), dim=1) # مشکل۵
    loss = criterion(probs, labels)
    loss.backward()

# گرادیان رو صفر نمی کنه و این باعث میشه گرادیان ها به صورت تجمیعی بیشتر و بیشتر میشه 
optimizer.step() # مشکل۶
(چیزی قبل ازاین خط جاافتاده)
final_model.eval()
with torch.no_grad():
    for images, labels in leaky_test_loader:
        predictions = final_model(images).argmax(dim=1)
#اینجا هم لاس ولیدیشن محاسبه نشده 
```
<br>
## نسخه درست کد :

```text
DEVICE = torch.device("cuda" if torch.cuda.is_available else "cpu")

TRAIN_LIMIT = 4000
TEST_LIMIT = 1000

final_model = TinyResNetFashion(num_classes=len(class_names).to(DEVICE)

train_aug_full = datasets.FashionMNIST(
root="data", train=True, download=True, transform=train_augmentation
)

train_loader = DataLoader(
Subset(train_aug_full,range(TRAIN_LIMIT)),batchsize=32,shuffle=True
)

test_full = datasets.FashionMNIST(
root="data",train=False,download=True,transform=baseline_transform
)

test_loader = DataLoader(
Subset(test_full, range(TEST_LIMIT)), batch_size=64, shuffle=False
)

optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

final_model.train()
for images, labels in train_loader:
    images,labels = images.to(DEVICE) ,labels.to(DEVICE
    logits = final_model(images)
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 
    
final_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images,labels = images.to(DEVICE) , labels.to(DEVICE)
        predictions = final_model(images).argmax(dim=1)
        
        
        
```
<br>

### خطاب به مدیر عامل :
در آموزش مدل ما از داده های ترین برای استراتژی های مثل آگمنتیشن و ... استفاده میکنیم تا مدل تصاویر متنوع ببیند ولی وقتی اقدام به تست مدل میکنیم باید یک مجموعه تست ثابت داشته باشیم تا آزمایش بین مدل ها معنا دار باشد ! اگر هر بار داده های تست شافل بخورد یا آگمنتیشن های تصادفی روی آن اعمال شود باعث میشود مجموعه تست قابل تکیه نباشد و نتایج  تصادفی با هر بار اجرای مدل تولید کند .

</div>