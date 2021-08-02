import * as tf from "@tensorflow/tfjs"
import * as tfvis from "@tensorflow/tfjs-vis"


window.onload = async () => {
    const heights = [150, 160, 170];
    const weights = [40, 50, 60];

    tfvis.render.scatterplot(
        { name: "身高体重线性关系预测" },
        { values: heights.map((x, i) => ({ x, y: weights[i] })) },
        { xAxisDomain: [140, 200], yAxisDomain: [30, 80] }
    )

    const inputs = tf.tensor(heights).sub(150).div(20)
    inputs.print()
    const outputs = tf.tensor(weights).sub(40).div(20)
    outputs.print()

    
    const model = tf.sequential();

    model.add(tf.layers.dense(
        { units: 1, inputShape: [1] },
    ))
    model.compile({ loss: tf.losses.meanSquaredError, optimizer: tf.train.sgd(0.1) })

    await model.fit(inputs, outputs, {
        batchSize: 3,
        epochs: 200,
        callbacks: tfvis.show.fitCallbacks(
            { name: "身高体重线性回归训练" },
            ['loss']
        )
    })
    const input=tf.tensor([parseInt(prompt("请输入待预测值"))]).sub(150).div(20)
    const output = model.predict(input);
    alert(`预期为${output.mul(20).add(40).dataSync()[0]}`);

    //http://localhost:1234








}


