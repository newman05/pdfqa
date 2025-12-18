import base64

def img_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

BOT_IMG = img_base64("images/ko.jpg")
USER_IMG = img_base64("images/ok.jpg")

css = '''
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
}
</style>
'''

bot_template = f'''
<div class="chat-message bot">
    <div class="avatar">
        <img src="data:image/jpeg;base64,{BOT_IMG}">
    </div>
    <div class="message">{{{{MSG}}}}</div>
</div>
'''

user_template = f'''
<div class="chat-message user">
    <div class="avatar">
        <img src="data:image/jpeg;base64,{USER_IMG}">
    </div>
    <div class="message">{{{{MSG}}}}</div>
</div>
'''
