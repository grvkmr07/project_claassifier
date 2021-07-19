/* Here we populate dataset with sample images and label them accordingly
 * xs contains tensor form of images
 * ys contains one-hot encoded form as labels
 */

class Dataset {
  constructor() {
    this.labels = []
  }
/*
 * addition of samples into the labels array
 */
  addExample(example, label) {
    // if array is empty
    if (this.xs == null) {
        // array holding tensor form  of the image captured
      this.xs = tf.keep(example);
        //push label into array
      this.labels.push(label);
    } // else if array is non-empty
      else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));
      this.labels.push(label);
      oldX.dispose();
    }
  }
    
  /*
   * mapping the tensors ofimages to one-hot encoding
   */
  encodeLabels(numClasses) {
    for (var i = 0; i < this.labels.length; i++) {
        // if array of ys is empty
      if (this.ys == null) {
          // one hot encoding as per labels
        this.ys = tf.keep(tf.tidy(
            () => {return tf.oneHot(
                tf.tensor1d([this.labels[i]]).toInt(), numClasses)}));
      } else {
          // one-hot encoding as per labels 
        const y = tf.tidy(
            () => {return tf.oneHot(
                tf.tensor1d([this.labels[i]]).toInt(), numClasses)});
        const oldY = this.ys;
        this.ys = tf.keep(oldY.concat(y, 0));
        oldY.dispose();
        y.dispose();
      }
    }
  }
}
