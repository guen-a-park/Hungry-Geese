# Hungry Geese
VGG, Googlenet 구조를 참고해 더 많은 layer를 이용해 더 deep한 모델을 사용해보려고 했다. 하지만 이 모델을 이용한 agent가 기존 코드에서 제공한 간단한 모델을 이용한 agent에 비해 performance가 좋지 않았다. 이런 결과를 바탕으로 비교적 간단한 task에 비해 너무 많은 parameter가 존재해 오히려 performance에 악영향을 끼친 것은 아닌지 하는 생각이 들어 주어진 모델을 더 단순화시켜 보았다.

1. main_1

def DQNet():

    model = tf.keras.Sequential()

    model.add(tf.keras.Input(shape=(7,11,17)))
    
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,5), activation='relu'))
    
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
    
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
    
    model.add(tf.keras.layers.Flatten())
    
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    
    model.add(tf.keras.layers.Dense(4, activation='linear'))
    
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    
    return model

Convolution layer는 동일하게 정의하고, fully connected layer의 개수를 줄이고 node의 수도 줄였다.

2. main_2

def DQNet():

    model = tf.keras.Sequential()
    
    model.add(tf.keras.Input(shape=(7,11,17)))
    
    #model.add(tf.keras.layers.UpSampling2D(size=(11, 7)))
    
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,5), activation='relu'))
    
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
    
    model.add(tf.keras.layers.Flatten())
    
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    
    model.add(tf.keras.layers.Dense(4, activation='linear'))
 
Convolution layer의 개수를 하나 줄였다. fully connected layer의 개수를 줄이고 node의 수도 줄였다.

Learning rate은 0.5, Discount factor는 0.6으로 수정해보았다.
두 모델을 비교했을 때, Convolution layer의 개수를 하나 줄인 두번째 모델이 더 좋은 performance를 보여 주었다.
