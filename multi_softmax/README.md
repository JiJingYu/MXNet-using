#multi_softmax用法

- 唯一要留意的是输入的数据格式和对应的labels格式
- 对于多输出情况，如输出labels为 (h, w) ， 类别数为c
- 那么softmax层的输入数据格式应为  (h, c, w)


- the only thing should be attention is the shape of input data and labels
- for multi-output, the shape of 'labels' is (h, w), and it has c classes
- so the shape of 'data' should be (h, c, w)
