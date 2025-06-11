module.exports = class CustomAPIProvider {
  id() {
    return "custom-api"; // Ensure this method correctly returns the provider ID.
  }

  async callApi(prompt) {
    const response = await fetch("http://127.0.0.1:8000/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: prompt }),
    });

    const data = await response.json();
    return { output: data.answer };
  }
};