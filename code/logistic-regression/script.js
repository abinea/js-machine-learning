import * as tf from "@tensorflow/tfjs"
import * as tfvis from "@tensorflow/tfjs-vis"
import { getData } from "./data"

window.onload = async () => {
    const data = getData(40)
    // console.log(data)
    tfvis.render.scatterplot(
        { name: "逻辑回归训练" },
        {
            values: [
                data.filter(p => p.label === 1),
                data.filter(p => p.label === 0),
            ]
        }
    )

    const model = tf.sequential()

    model.add(tf.layers.dense({
        units: 1,
        inputShape: [2],
        activation: 'sigmoid'
    }))
    model.compile({ loss: tf.losses.logLoss, optimizer: tf.train.adam(0.1) })

    const inputs = tf.tensor(data.map(p => ([p.x, p.y])))
    const labels = tf.tensor(data.map(p => p.label))

    await model.fit(inputs, labels, {
        batchSize: 40,
        epochs: 50,
        callbacks: tfvis.show.fitCallbacks(
            { name: "实例化逻辑回归" },
            ['loss']
        )
    })

    window.predict = (form) => {
        const pred = model.predict(tf.tensor([[form.x.value * 1, form.y.value * 1]]))
        alert(`预期为${pred.dataSync()[0]}`);
    }

}