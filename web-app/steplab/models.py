from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

class Post(models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField()
    date_posted = models.DateField(default=timezone.now)
    author = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.title

class StepFile(models.Model):

    title = models.CharField(max_length=512)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    footsize = models.IntegerField()
    productId = models.CharField(max_length=64)
    steps = models.IntegerField()
    content = models.TextField()

    def __str__(self):
        return '{} {} {} {} {}'.format(self.title, self.author, self.footsize, self.productId, self.steps)


class StepPrediction(models.Model):
    user =  models.ForeignKey(User, on_delete=models.CASCADE)
    files = models.TextField()
    creationDate = models.DateTimeField(auto_now_add=True)


class StepGroup(models.Model):
    '''
        https://docs.djangoproject.com/en/2.2/ref/models/fields/#django.db.models.Field.primary_key
    '''
    stepPrediction =  models.ForeignKey(StepPrediction, on_delete=models.CASCADE)
    groupIndex = models.IntegerField()
    originIndex = models.IntegerField()
    endIndex = models.IntegerField()
    size = models.IntegerField()

class StepGroupClassiffier(models.Model):
    stepGroup = models.ForeignKey(StepGroup, on_delete=models.CASCADE)
    goodSteps = models.IntegerField(default=0)
    badSteps = models.IntegerField(default=0)
    riskFalling = models.BooleanField()


class StepSession(models.Model):
    name = models.TextField(max_length=512)

    date_uploaded = models.DateTimeField(auto_now_add=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE)

'''
    Wrapper class for all the values that are generated on one step.

    App_time;Step_number;Insole_timer;Contact_time;ESW_timer;S0_force;S0_start_time;
    S0_max_time;S0_end_time;S1_force;S1_start_time;S1_max_time;S1_end_time;
    S2_force;S2_start_time;S2_max_time;S2_end_time;S3_force;S3_start_time;S3_max_time;S3_end_time;
    S4_force;S4_start_time;S4_max_time;S4_end_time;S5_force;S5_start_time;S5_max_time;S5_end_time;
    S6_force;S6_start_time;S6_max_time;S6_end_time;F1_force;F1_time;F2_force;F2_time;F3_force;
    F3_time;Warning_code;Lifetime_steps;Day;Month;Year;Left/Right;Size;Insole_id;Battery;MAC
'''
'''    
class StepRecord(models.Model):
    

    # user_id = 

    App_time = models.DateField()
    Step_number = models.IntegerField()
    Insole_timer = models.IntegerField()
    Contact_time = models.IntegerField()
    ESW_timer = models.IntegerField()
    S0_force = models.IntegerField()
    S0_start_time = models.IntegerField()
    S0_max_time = models.IntegerField()
    S0_end_time = models.IntegerField()
    S1_force = models.IntegerField()
    S1_start_time = models.IntegerField()
    S1_max_time = models.IntegerField()
    S1_end_time = models.IntegerField()
    S2_force = models.IntegerField()
    S2_start_time = models.IntegerField()
    S2_max_time = models.IntegerField()
    S2_end_time = models.IntegerField()
    S3_force = models.IntegerField()
    S3_start_time = models.IntegerField()
    S3_max_time = models.IntegerField()
    S3_end_time = models.IntegerField()
    S4_force = models.IntegerField()
    S4_start_time = models.IntegerField()
    S4_max_time = models.IntegerField()
    S4_end_time = models.IntegerField()
    S5_force = models.IntegerField()
    S5_start_time = models.IntegerField()
    S5_max_time = models.IntegerField()
    S5_end_time = models.IntegerField()
    S6_force = models.IntegerField()
    S6_start_time = models.IntegerField()
    S6_max_time = models.IntegerField()
    S6_end_time = models.IntegerField()
    F1_force = models.IntegerField()
    F1_time = models.IntegerField()
    F2_force = models.IntegerField()
    F2_time = models.IntegerField()
    F3_force = models.IntegerField()
    F3_time = models.IntegerField()
    Warning_code = models.IntegerField()
    Lifetime_steps = models.IntegerField()
    DayMonth = models.IntegerField()
    Year = models.IntegerField()
    Left/Right = models.CharField(max_length=1)
    Size = models.IntegerField()
    Insole_id = models.IntegerField()
    Battery = models.IntegerField()
    MAC = models.TextField()

    def __str__(self):
        return  self.App_time + self.Step_number + self.Insole_timer + self.Contact_time + self.ESW_timer + self.S0_force + self.S0_start_time +
                self.S0_max_time + self.S0_end_time + self.S1_force + self.S1_start_time + self.S1_max_time + self.S1_end_time +
                self.S2_force + self.S2_start_time + self.S2_max_time + self.S2_end_time + self.S3_force + self.S3_start_time + self.S3_max_time + self.S3_end_time +
                self.S4_force + self.S4_start_time + self.S4_max_time + self.S4_end_time + self.S5_force + self.S5_start_time + self.S5_max_time + self.S5_end_time +
                self.S6_force + self.S6_start_time + self.S6_max_time + self.S6_end_time + self.F1_force + self.F1_time + self.F2_force + self.F2_time + self.F3_force +
                self.F3_time + self.Warning_code + self.Lifetime_steps + self.Day + self.Month + self.Year + self.Left/Right + self.Size + self.Insole_id + self.Battery

'''