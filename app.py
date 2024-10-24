import tensorflow as tf

# 텐서플로우에서 제공하는 텐서라는 자료형이 있음
# 행렬, 벡터  -> 예측결과값 계산, 레이어 생성, w값 갱신
# learning rate 생성, 손실함수 등등을 쉽게 사용할 수 있음

#constant는 상수
tensor = tf.constant([1,2,3])
# print(tensor)

tensor1 = tf.constant([1,2,3])
tensor2 = tf.constant([10,20,30])

#print(tensor1 + tensor2)
#print(tf.add(tensor1,tensor2)) #더하기
#print(tf.subtract(tensor1,tensor2)) #빼기
#print(tf.divide(tensor1,tensor2)) # 나누기
#print(tf.multiply(tensor1,tensor2)) # 곱하기

tensor3 = tf.constant([[1,2],[3,4]])
#print(tensor3)

# print(tf.matmul(tensor1,tensor2)) 1차원은 계산안됨
tensor1 = tf.constant([[1,2,3]])
tensor2 = tf.constant([[10,20,30]])
# print(tf.matmul(tensor1,tensor2)) 2차원이지만 가로라 안됨?
# tensor2를 행 3개로 만들어 곱하니 됨
#print(tf.matmul(tensor1,tf.transpose(tensor2)))

tensor1 = tf.constant([[1,2],[3,4]])
tensor2 = tf.constant([[10,20],[30,40]])

#print(tf.matmul(tensor1,tensor2))


tensor1 = tf.constant([[1,2],[3,4],[5,6]]) # 3*2
tensor2 = tf.constant([[10,20,30],[30,40,50]]) # 2*3
# print(tf.matmul(tensor1,tensor2))


t = tf.zeros(10)
#print(t)
t2 = tf.zeros([3,2])
#print(t2)
t3 = tf.zeros([3,2,5])
#print(t3)

#print(t3.shape)
#print(t3.dtype)

v1 = tf.Variable(1)
v2 = tf.Variable([1,2,3])

v1.assign(100)
#값을 바꿀 수 있는 변수 텐서
#print(v1.numpy())
#print(v2)

v2.assign([10,20,30])
#print(v2)
d =v2.numpy()
d[0] = 1000
v2.assign(d)
print(v2)

v2.scatter_nd_update(indices=[[1]], updates=[500])
print(v2)

