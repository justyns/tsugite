function quitHandler({ supervisor, exit, onFailed }) {
  let exiting = false;
  let pending = false;
  return (event) => {
    if (exiting) return;
    event.preventDefault();
    if (pending) return;
    pending = true;
    supervisor.quit().then(
      () => {
        exiting = true;
        exit();
      },
      (err) => {
        pending = false;
        onFailed(err);
      },
    );
  };
}

module.exports = { quitHandler };