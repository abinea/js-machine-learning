import * as tf from "@tensorflow/tfjs"
import * as tfvis from "@tensorflow/tfjs-vis"
//保证所有的依赖加载完再运行代码
window.onload = async () => {
    const xs = [1, 2, 3, 4];
    const ys = [1, 3, 5, 7]

    // tfvis.render.scatterplot(
    //     { name: "linear-regression" },
    //     { values: xs.map((x, i) => ({ x, y: ys[i] })) },
    //     { xAxisDomain: [0, 6], yAxisDomain: [0, 10] }
    // )

    const model = tf.sequential();

    model.add(tf.layers.dense({
        units: 1,
        inputShape: [1]
    }))
    model.compile({ loss: tf.losses.meanSquaredError, optimizer: tf.train.sgd(0.1) })

    const inputs = tf.tensor(xs)
    const labels = tf.tensor(ys)
    await model.fit(inputs, labels, {
        batchSize: 4,
        epochs: 100,
        callbacks: tfvis.show.fitCallbacks(
            { name: "线性回归训练" },
            ['loss']
        )
    })

    const input = parseInt(prompt("请输入待预测值"))
    const output = model.predict(tf.tensor([input]));
    // output.print();
    alert(`预期为${output.dataSync()[0]}`);




}