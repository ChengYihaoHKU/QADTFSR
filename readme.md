# Quality-Assisted Domain Transfer for Fast Face Super-Resolution
>Yi-Hao Cheng, Wan-Chi Siu (Life-FIEEE), and Shing-Chow Chan (SrMIEEE)
## About
**This Github document is to demonstrate more visual results of proposed method.**

---
### A. Synthetic CelebA dataset.
![Visual Comparison with State-of-the-Art Methods](images/comparsion/fig1.png)
Fig. 1. Visual comparison with state-of-the-art methods on the Synthetic CelebA Dataset. For methods involving iterative sampling, we use the notation “Method-x”, where x denotes the number of sampling steps.

![Visual Comparison with State-of-the-Art Methods](images/comparsion/fig2.png)
Fig. 2. Visual examples of our method.
### B. Commonly used real-world datasets.
![Visual Comparison with State-of-the-Art Methods](images/comparsion/fig3.png)
Fig. 2. Visual comparison with state-of-the-art methods. For methods involving iterative sampling, we use the notation “Method-x”, where x denotes the number of sampling steps.
<!DOCTYPE html>
<html lang="en">
<head>
  <script type="module" src="https://unpkg.com/img-comparison-slider@8/dist/index.js"></script>
  <link rel="stylesheet" href="https://unpkg.com/img-comparison-slider@8/dist/styles.css">
  <style>
    .label {
      position: absolute;
      top: 10px;
      padding: 8px 16px;
      border-radius: 5px;
      font-weight: bold;
      font-size: 14px;
      color: rgba(255, 255, 255, 1);
      z-index: 10;
    }
    .label-left {
      left: 10px;
    }
    .label-right {
      right: 10px;
    }
  </style>
</head>
<body>
  <div class="test-section">
    <img-comparison-slider>
      <img slot="first" src="images/real-world/lfw/Aileen_Riggin_Soule_0001_00.png" alt="Before" />
      <img slot="second" src="images/real-world/lfw/sr_Aileen_Riggin_Soule_0001_00.png" alt="After" />
      <div slot="first" class="label label-left">
        Upsampled LR
      </div>
      <div slot="second" class="label label-right">
        SR Image
      </div>
    </img-comparison-slider>
    <p>Fig. 3. Comparison between Upsampled LR and SR Image on real-world LR dataset LFW.</p>
    <br>
  </div>
  <div class="test-section">
    <img-comparison-slider>
      <img slot="first" src="images/real-world/wider/0035.png" alt="Before" />
      <img slot="second" src="images/real-world/wider/sr_0035.png" alt="After" />
      <div slot="first" class="label label-left">
        Upsampled LR
      </div>
      <div slot="second" class="label label-right">
        SR Image
      </div>
    </img-comparison-slider>
    <p>Fig. 4. Comparison between Upsampled LR and SR Image on real-world LR dataset Wider-Test.</p>
    <br>
  </div>
  <div class="test-section">
    <img-comparison-slider>
      <img slot="first" src="images/real-world/webphoto/0034.png" alt="Before" />
      <img slot="second" src="images/real-world/webphoto/sr_0034.png" alt="After" />
      <div slot="first" class="label label-left">
        Upsampled LR
      </div>
      <div slot="second" class="label label-right">
        SR Image
      </div>
    </img-comparison-slider>
    <p>Fig. 5. Comparison between Upsampled LR and SR Image on real-world LR dataset WebPhoto-Test.</p>
  </div>
</body>
</html>

## C. Natural LR faces extracted from a Classroom Photo.
![classroom](images/real-world/classroom/classroom.png)
Fig. 6. Students in classroom. Original image resolution: 1440*960. Image from [1]. X: Y means X is the face index, Y is the resolution.

![classroom](images/real-world/classroom/result.png)
Fig. 7. Visual comparison with state-of-the-art methods on natural LR faces extracted from a classroom photo.

### Reference
[1] D. Newton, "Online college classes should have no more than 12 students," Forbes, Jun. 28, 2020. [Online]. Available: https://www.forbes.com/sites/dereknewton/2020/06/28/online-college-classes-should-have-no-more-than-12-students/. [Accessed: Nov. 24, 2025].

